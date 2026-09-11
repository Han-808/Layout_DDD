/*
 * Import-map shim for module-loaded three.js.
 *
 * The rewritten entry document points the bare `three` specifier here and moves
 * the page's original three URL to a private specifier, so the game still runs
 * against the exact build it shipped with. This module re-exports that build
 * unchanged except for `Scene` and `WebGLRenderer`. Scene registers every
 * instance; WebGLRenderer records the exact scene/camera pairs rendered by the
 * authored program.  The latter lets the benchmark render another camera
 * through the original runtime without reconstructing appearance.
 *
 * `export *` skips names that are also exported explicitly, so the local `Scene`
 * below shadows the original without needing to enumerate the rest of the API.
 * Subclassing keeps `isScene`, `instanceof`, and the prototype chain intact, so
 * addons that branch on any of those behave identically.
 */
import * as THREE from "__THREE_SOURCE__";

if (!window.__BENCHMARK_SCENES__) {
  window.__BENCHMARK_SCENES__ = [];
}
window.__BENCHMARK_THREE__ = THREE;

class Scene extends THREE.Scene {
  constructor() {
    super(...arguments);
    if (typeof window.__benchmarkRegisterScene === "function") {
      window.__benchmarkRegisterScene(this);
    } else {
      window.__BENCHMARK_SCENES__.push(this);
    }
  }
}

class WebGLRenderer extends THREE.WebGLRenderer {
  constructor() {
    super(...arguments);
    if (typeof window.__benchmarkRegisterRenderer === "function") {
      window.__benchmarkRegisterRenderer(this);
    }
  }
  render(scene, camera) {
    if (typeof window.__benchmarkRegisterRender === "function") {
      window.__benchmarkRegisterRender(this, scene, camera);
    }
    return super.render(...arguments);
  }
}

try {
  Object.defineProperty(Scene, "name", { value: "Scene" });
  Object.defineProperty(WebGLRenderer, "name", { value: "WebGLRenderer" });
} catch (error) {
  /* Non-configurable name is harmless; the prototype chain still matches. */
}

export * from "__THREE_SOURCE__";
export { Scene, WebGLRenderer };
