/*
 * Render one benchmark-owned camera through the original Three.js runtime.
 *
 * The load-time harness records direct WebGLRenderer.render(scene, camera)
 * calls. This function reuses that exact renderer and scene, creates only an
 * ephemeral camera, and leaves the resulting pixels on the renderer canvas for
 * Playwright to capture. The authored camera and simulation state are never
 * mutated.
 */
(function renderBenchmarkCameraView(options) {
  var config = options || {};
  var THREE = window.__BENCHMARK_THREE__ || window.THREE;
  var registry = window.__BENCHMARK_RENDERS__ || [];

  function fail(code, detail) {
    return { ok: false, code: code, detail: String(detail || "") };
  }

  if (!THREE || typeof THREE.PerspectiveCamera !== "function") {
    return fail("three_namespace_unavailable");
  }
  if (!registry.length) {
    return fail("direct_webgl_render_context_unavailable");
  }
  if (registry.length !== 1) {
    return fail("ambiguous_multiple_webgl_renderers", registry.length);
  }

  var scenes = window.__BENCHMARK_SCENES__ || [];
  var richestScene = null;
  for (var sceneIndex = 0; sceneIndex < scenes.length; sceneIndex += 1) {
    var registeredScene = scenes[sceneIndex];
    if (!registeredScene || !registeredScene.isScene) {
      continue;
    }
    if (
      richestScene === null ||
      (registeredScene.children || []).length > (richestScene.children || []).length
    ) {
      richestScene = registeredScene;
    }
  }

  var record = null;
  for (var i = 0; i < registry.length; i += 1) {
    var candidate = registry[i];
    if (!candidate || !candidate.renderer) {
      continue;
    }
    var candidateScene = candidate.scene || richestScene;
    if (!candidateScene) {
      continue;
    }
    if (
      record === null ||
      (candidateScene.children || []).length >
        ((record.scene || richestScene).children || []).length
    ) {
      record = candidate;
    }
  }
  if (record === null) {
    return fail("direct_webgl_render_context_unavailable");
  }

  var renderer = record.renderer;
  var scene = record.scene || richestScene;
  var canvas = renderer.domElement;
  if (!canvas || !canvas.isConnected || String(canvas.tagName).toLowerCase() !== "canvas") {
    return fail("connected_renderer_canvas_unavailable");
  }
  if (typeof renderer.render !== "function") {
    return fail("direct_webgl_render_unavailable");
  }

  var pose = config.pose || {};
  var location = pose.location;
  var target = pose.target;
  if (!Array.isArray(location) || location.length !== 3) {
    return fail("invalid_camera_location");
  }
  if (!Array.isArray(target) || target.length !== 3) {
    return fail("invalid_camera_target");
  }

  var width = Math.max(1, Number(canvas.width) || Number(canvas.clientWidth) || 1);
  var height = Math.max(1, Number(canvas.height) || Number(canvas.clientHeight) || 1);
  var aspect = width / height;
  var camera;
  if (pose.camera_type === "ORTHO") {
    var scale = Number(pose.ortho_scale);
    if (!isFinite(scale) || scale <= 0) {
      return fail("invalid_ortho_scale");
    }
    camera = new THREE.OrthographicCamera(
      -scale * aspect / 2,
      scale * aspect / 2,
      scale / 2,
      -scale / 2,
      Number(pose.near) || 0.01,
      Number(pose.far) || 10000
    );
  } else {
    var fov = Number(pose.vertical_fov_degrees);
    if (!isFinite(fov) || fov <= 1 || fov >= 179) {
      return fail("invalid_vertical_fov");
    }
    camera = new THREE.PerspectiveCamera(
      fov,
      aspect,
      Number(pose.near) || 0.01,
      Number(pose.far) || 10000
    );
  }

  camera.position.set(Number(location[0]), Number(location[1]), Number(location[2]));
  var up = Array.isArray(pose.up) && pose.up.length === 3 ? pose.up : [0, 1, 0];
  camera.up.set(Number(up[0]), Number(up[1]), Number(up[2]));
  camera.lookAt(Number(target[0]), Number(target[1]), Number(target[2]));
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  scene.updateMatrixWorld(true);

  var previousTarget =
    typeof renderer.getRenderTarget === "function" ? renderer.getRenderTarget() : null;
  var previousScissorTest =
    typeof renderer.getScissorTest === "function" ? renderer.getScissorTest() : false;
  var previousAutoClear = renderer.autoClear;
  var suppressedRuntimeNodes = [];
  var excludedRuntimeRoles = {
    dynamic_actor: true,
    dynamic_player: true,
    projectile: true,
    effect: true,
    transient_helper: true,
    helper: true,
    viewmodel: true,
  };
  /*
   * The benchmark scores the frozen static environment, not the bots captured
   * at a particular deterministic tick.  ``probe.js`` has already classified
   * every visible mesh using the same runtime-role contract that filtered the
   * canonical/collision outputs.  Temporarily hide those exact meshes for the
   * controlled views, then restore their authored visibility after readback.
   */
  if (typeof scene.traverse === "function") {
    scene.traverse(function suppressRuntimeEntity(node) {
      var role = node && node.userData && node.userData.__benchmarkRuntimeRole;
      if (!excludedRuntimeRoles[String(role || "")]) {
        return;
      }
      suppressedRuntimeNodes.push({ node: node, visible: node.visible });
      node.visible = false;
    });
  }
  var viewport = typeof THREE.Vector4 === "function" ? new THREE.Vector4() : null;
  var scissor = typeof THREE.Vector4 === "function" ? new THREE.Vector4() : null;
  if (viewport && typeof renderer.getViewport === "function") {
    renderer.getViewport(viewport);
  }
  if (scissor && typeof renderer.getScissor === "function") {
    renderer.getScissor(scissor);
  }

  var viewId = String(config.view_id || "controlled_global");
  try {
    window.__BENCHMARK_CONTROLLED_RENDER_ACTIVE__ = true;
    if (typeof renderer.setRenderTarget === "function") {
      renderer.setRenderTarget(null);
    }
    if (typeof renderer.setViewport === "function") {
      renderer.setViewport(0, 0, width, height);
    }
    if (typeof renderer.setScissorTest === "function") {
      renderer.setScissorTest(false);
    }
    renderer.autoClear = true;
    renderer.render(scene, camera);
    canvas.setAttribute("data-benchmark-controlled-view", viewId);
  } catch (error) {
    return fail("controlled_camera_render_failed", error && error.name);
  } finally {
    window.__BENCHMARK_CONTROLLED_RENDER_ACTIVE__ = false;
    renderer.autoClear = previousAutoClear;
    if (typeof renderer.setRenderTarget === "function") {
      renderer.setRenderTarget(previousTarget);
    }
    if (viewport && typeof renderer.setViewport === "function") {
      renderer.setViewport(viewport);
    }
    if (scissor && typeof renderer.setScissor === "function") {
      renderer.setScissor(scissor);
    }
    if (typeof renderer.setScissorTest === "function") {
      renderer.setScissorTest(previousScissorTest);
    }
    for (var restoreIndex = 0; restoreIndex < suppressedRuntimeNodes.length; restoreIndex += 1) {
      var saved = suppressedRuntimeNodes[restoreIndex];
      saved.node.visible = saved.visible;
    }
  }

  var imageDataUrl;
  try {
    imageDataUrl = canvas.toDataURL("image/png");
  } catch (error) {
    return fail("controlled_canvas_readback_failed", error && error.name);
  }
  if (typeof imageDataUrl !== "string" || imageDataUrl.indexOf("data:image/png;base64,") !== 0) {
    return fail("controlled_canvas_readback_invalid");
  }
  return {
    ok: true,
    code: "original_runtime_direct_webgl",
    view_id: viewId,
    image_data_url: imageDataUrl,
    canvas_width: width,
    canvas_height: height,
    registered_renderer_count: registry.length,
    direct_render_call_count: Number(record.directRenderCallCount) || 0,
    suppressed_runtime_mesh_count: suppressedRuntimeNodes.length,
    appearance_fidelity: "original_runtime_direct_webgl",
    render_context_source:
      record.scene === null ? "renderer_constructor_plus_scene_registry" : "authored_render_call",
  };
})
