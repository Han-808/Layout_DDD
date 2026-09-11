/*
 * Load-time scene instrumentation, injected ahead of every page script.
 *
 * Searching `window` for a live scene after the fact does not work on real
 * games: a classic script's top-level `const` never becomes a window property,
 * and a scene held as a class field is closed over. The only reliable capture
 * point is construction, so this harness wraps the `Scene` constructor and
 * records every instance in `window.__BENCHMARK_SCENES__`.
 *
 * The same load-time seam records direct WebGLRenderer calls.  This is the
 * renderer-level contract used by benchmark-controlled cameras: it preserves
 * the original scene, materials, textures, lights, fog, shaders, and renderer
 * settings instead of reconstructing the game in Blender.  The hook records
 * context only; it never changes the authored render call.
 *
 * Two delivery paths exist because three.js arrives two ways. Script-tag games
 * get the trap installed here, on the `window.THREE` namespace. Module games get
 * it from the import-map shim, which calls `__benchmarkRegisterScene` instead.
 * Both converge on the same registry.
 *
 * The `window.THREE` case needs care: three's UMD wrapper assigns an empty
 * object first and populates it afterwards, so trapping the assignment alone
 * would see a namespace with no `Scene` on it. The trap is therefore reinstalled
 * as an accessor on the namespace object itself.
 */
(function installSceneInstrumentation(options) {
  var config = options || {};
  var registry = window.__BENCHMARK_SCENES__ || [];
  var renderRegistry = window.__BENCHMARK_RENDERS__ || [];
  var shadowedNativeNamespaces = [];
  window.__BENCHMARK_SCENES__ = registry;
  window.__BENCHMARK_RENDERS__ = renderRegistry;

  /*
   * Generated classic-script games occasionally publish a level namespace as
   * `window.Map`. That replaces the ECMAScript Map constructor and crashes
   * Three.js itself when WebGLRenderer later executes `new Map()`. Preserve the
   * native constructor while accepting the generated namespace's public fields,
   * so both `new Map()` and `Map.build(scene)` retain their intended meanings.
   * This is a browser-runtime guard, not a per-scene rewrite.
   */
  function protectNativeConstructorNamespace(name) {
    var nativeConstructor = window[name];
    if (typeof nativeConstructor !== "function") {
      return;
    }
    try {
      Object.defineProperty(window, name, {
        configurable: true,
        enumerable: false,
        get: function getProtectedNative() {
          return nativeConstructor;
        },
        set: function publishNamespace(value) {
          if (
            value === nativeConstructor ||
            value === null ||
            (typeof value !== "object" && typeof value !== "function")
          ) {
            return;
          }
          Object.keys(value).forEach(function copyNamespaceField(key) {
            try {
              nativeConstructor[key] = value[key];
            } catch (error) {
              /* A reserved constructor field stays native. */
            }
          });
          if (shadowedNativeNamespaces.indexOf(name) === -1) {
            shadowedNativeNamespaces.push(name);
          }
        },
      });
    } catch (error) {
      /* A locked global remains untouched and the source keeps its own fate. */
    }
  }
  protectNativeConstructorNamespace("Map");

  function register(scene) {
    if (scene && registry.indexOf(scene) === -1) {
      registry.push(scene);
    }
    return scene;
  }
  window.__benchmarkRegisterScene = register;

  function registerRenderer(renderer) {
    if (!renderer) {
      return null;
    }
    for (var i = 0; i < renderRegistry.length; i += 1) {
      if (renderRegistry[i].renderer === renderer) {
        return renderRegistry[i];
      }
    }
    var record = {
      renderer: renderer,
      scene: null,
      authoredCamera: null,
      lastCamera: null,
      directRenderCallCount: 0,
    };
    renderRegistry.push(record);
    return record;
  }
  window.__benchmarkRegisterRenderer = registerRenderer;

  function registerRender(renderer, scene, camera) {
    if (!renderer || !scene || !camera) {
      return null;
    }
    var record = registerRenderer(renderer);
    record.scene = scene;
    record.lastCamera = camera;
    record.directRenderCallCount += 1;
    if (!window.__BENCHMARK_CONTROLLED_RENDER_ACTIVE__) {
      record.authoredCamera = camera;
    }
    return record;
  }
  window.__benchmarkRegisterRender = registerRender;

  function wrapSceneConstructor(RealScene) {
    if (typeof RealScene !== "function" || RealScene.__benchmarkWrapped) {
      return RealScene;
    }
    class BenchmarkScene extends RealScene {
      constructor() {
        super(...arguments);
        register(this);
      }
    }
    // Games and libraries occasionally branch on the constructor name, so the
    // subclass must be indistinguishable from the original by that test.
    try {
      Object.defineProperty(BenchmarkScene, "name", { value: RealScene.name });
    } catch (error) {
      /* Non-configurable name is harmless; the prototype chain still matches. */
    }
    BenchmarkScene.__benchmarkWrapped = true;
    BenchmarkScene.__benchmarkRealScene = RealScene;
    return BenchmarkScene;
  }

  function wrapRendererConstructor(RealRenderer) {
    if (typeof RealRenderer !== "function" || RealRenderer.__benchmarkWrapped) {
      return RealRenderer;
    }
    class BenchmarkWebGLRenderer extends RealRenderer {
      constructor() {
        super(...arguments);
        registerRenderer(this);
      }
      render(scene, camera) {
        registerRender(this, scene, camera);
        return super.render(...arguments);
      }
    }
    try {
      Object.defineProperty(BenchmarkWebGLRenderer, "name", { value: RealRenderer.name });
    } catch (error) {
      /* The prototype contract is sufficient when name is non-configurable. */
    }
    BenchmarkWebGLRenderer.__benchmarkWrapped = true;
    BenchmarkWebGLRenderer.__benchmarkRealRenderer = RealRenderer;
    return BenchmarkWebGLRenderer;
  }

  function trapConstructor(namespace, key, wrap) {
    var descriptor = Object.getOwnPropertyDescriptor(namespace, key);
    var wrapped = descriptor ? wrap(descriptor.value) : undefined;
    try {
      Object.defineProperty(namespace, key, {
        configurable: true,
        enumerable: true,
        get: function getConstructor() {
          return wrapped;
        },
        set: function setConstructor(value) {
          wrapped = wrap(value);
        },
      });
    } catch (error) {
      /* A sealed namespace cannot be trapped; module games use the shim. */
    }
  }

  function trapNamespace(namespace) {
    if (!namespace || typeof namespace !== "object" || namespace.__benchmarkTrapped) {
      return namespace;
    }
    try {
      Object.defineProperty(namespace, "__benchmarkTrapped", { value: true });
    } catch (error) {
      return namespace;
    }
    trapConstructor(namespace, "Scene", wrapSceneConstructor);
    trapConstructor(namespace, "WebGLRenderer", wrapRendererConstructor);
    window.__BENCHMARK_THREE__ = namespace;
    return namespace;
  }
  window.__benchmarkTrapThreeNamespace = trapNamespace;

  var existing = Object.getOwnPropertyDescriptor(window, "THREE");
  if (existing && existing.value) {
    trapNamespace(existing.value);
  } else {
    var stored;
    try {
      Object.defineProperty(window, "THREE", {
        configurable: true,
        enumerable: true,
        get: function getThree() {
          return stored;
        },
        set: function setThree(value) {
          stored = trapNamespace(value);
        },
      });
    } catch (error) {
      /* Nothing to trap on this page; the module shim path still applies. */
    }
  }

  window.__benchmarkInstrumentation = {
    installed: true,
    seed: config.seed === undefined ? null : config.seed,
    controlledCameraContextCapture: "direct_webgl_renderer_v1",
    shadowedNativeNamespaces: shadowedNativeNamespaces,
  };
})
