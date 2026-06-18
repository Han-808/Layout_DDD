import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/controls/OrbitControls.js";

const viewport = document.getElementById("viewport");
const titleEl = document.getElementById("task-title");
const iterationSelect = document.getElementById("iteration-select");
const beforeButton = document.getElementById("before-button");
const afterButton = document.getElementById("after-button");
const replayButton = document.getElementById("replay-button");
const summaryEl = document.getElementById("summary");
const workflowEl = document.getElementById("workflow");
const artifactTitleEl = document.getElementById("artifact-title");
const artifactPreviewEl = document.getElementById("artifact-preview");
const violationsEl = document.getElementById("violations");
const feedbackEl = document.getElementById("feedback");
const metricsEl = document.getElementById("metrics");
const roomViewsEl = document.getElementById("room-views");
const pairViewsEl = document.getElementById("pair-views");
const workflowTraceEl = document.getElementById("workflow-trace");
const relationsEl = document.getElementById("relations");
const historyEl = document.getElementById("history");

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0xeef0eb, 1);
viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeef0eb);

const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 500);
camera.position.set(5.5, 6.0, 6.5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(2.5, 0.5, 2.0);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xffffff, 0xb8b8a8, 1.8));
const directional = new THREE.DirectionalLight(0xffffff, 1.6);
directional.position.set(4, 8, 5);
scene.add(directional);

const sceneGroup = new THREE.Group();
scene.add(sceneGroup);

let viewerData = null;
let activeScene = null;
let selectedArtifactIndex = 0;
let highlightedObjectIds = new Set();
let replayTimer = null;

fetch("./viewer_scene.json", { cache: "no-store" })
  .then((response) => {
    if (!response.ok) {
      throw new Error(`Unable to load viewer_scene.json: ${response.status}`);
    }
    return response.json();
  })
  .then((data) => {
    viewerData = data;
    initSceneSelection(data);
    selectArtifact(0, { syncScene: true });
    renderActiveScene();
    window.__viewerReady = true;
  })
  .catch((error) => {
    window.__viewerError = error.message;
    titleEl.textContent = "Unable to load viewer_scene.json";
    violationsEl.innerHTML = `<div class="item invalid">${escapeHtml(error.message)}</div>`;
  });

iterationSelect.addEventListener("change", () => {
  stopReplay();
  const scenes = getScenes(viewerData);
  activeScene = scenes[Number(iterationSelect.value)];
  selectedArtifactIndex = findArtifactIndexForIteration(activeScene?.iteration);
  highlightedObjectIds = new Set();
  renderActiveScene();
  updateArtifactPreview(getArtifacts()[selectedArtifactIndex]);
});

beforeButton.addEventListener("click", () => {
  stopReplay();
  selectArtifact(findArtifactByStep("evaluate", 0) ?? 1, { syncScene: true });
});

afterButton.addEventListener("click", () => {
  stopReplay();
  selectArtifact(findArtifactByStep("evaluate", 1) ?? 5, { syncScene: true });
});

replayButton.addEventListener("click", () => {
  if (replayTimer) {
    stopReplay();
    return;
  }
  replayWorkflow();
});

window.addEventListener("resize", resize);
resize();
animate();

function initSceneSelection(data) {
  const scenes = getScenes(data);
  iterationSelect.innerHTML = "";
  scenes.forEach((item, index) => {
    const option = document.createElement("option");
    const status = item.overall_valid ? "valid" : "invalid";
    option.value = String(index);
    option.textContent = `iteration ${item.iteration ?? index} - ${status}`;
    iterationSelect.appendChild(option);
  });

  const firstInvalid = scenes.findIndex((item) => item.overall_valid === false);
  const selected = firstInvalid >= 0 ? firstInvalid : scenes.length - 1;
  iterationSelect.value = String(Math.max(0, selected));
  activeScene = scenes[Math.max(0, selected)];
}

function getScenes(data) {
  if (Array.isArray(data?.iterations) && data.iterations.length > 0) {
    return data.iterations;
  }
  return [data];
}

function getArtifacts() {
  return viewerData?.workflow?.artifacts || [];
}

function findSceneByIteration(iteration) {
  const scenes = getScenes(viewerData);
  return scenes.find((item) => item.iteration === iteration) || scenes[0];
}

function findArtifactIndexForIteration(iteration) {
  const artifacts = getArtifacts();
  const index = artifacts.findIndex((item) => item.iteration === iteration && item.step === "evaluate");
  if (index >= 0) return index;
  const fallback = artifacts.findIndex((item) => item.iteration === iteration);
  return fallback >= 0 ? fallback : selectedArtifactIndex;
}

function findArtifactByStep(step, iteration) {
  const artifacts = getArtifacts();
  const index = artifacts.findIndex((item) => item.step === step && item.iteration === iteration);
  return index >= 0 ? index : null;
}

function selectArtifact(index, options = {}) {
  const artifacts = getArtifacts();
  if (!artifacts.length) return;
  const boundedIndex = Math.max(0, Math.min(index, artifacts.length - 1));
  selectedArtifactIndex = boundedIndex;
  const artifact = artifacts[boundedIndex];

  if (options.syncScene && Number.isInteger(artifact.iteration)) {
    activeScene = findSceneByIteration(artifact.iteration);
    const scenes = getScenes(viewerData);
    const sceneIndex = scenes.findIndex((item) => item.iteration === artifact.iteration);
    if (sceneIndex >= 0) {
      iterationSelect.value = String(sceneIndex);
    }
  } else if (options.syncScene && artifact.step === "metrics") {
    activeScene = findSceneByIteration(1);
    const sceneIndex = getScenes(viewerData).findIndex((item) => item.iteration === 1);
    iterationSelect.value = String(Math.max(0, sceneIndex));
  }

  highlightedObjectIds = new Set();
  renderActiveScene();
  updateArtifactPreview(artifact);
}

function renderActiveScene() {
  if (!activeScene) return;
  clearGroup(sceneGroup);
  titleEl.textContent = activeScene.task_id || "viewer_scene.json";

  drawRoom(activeScene);
  drawObjects(activeScene);
  drawRelations(activeScene);
  updatePanel(activeScene, viewerData || {}, viewerData?.history || activeScene.history || []);
  frameScene(activeScene);
}

function drawRoom(data) {
  const room = data.room || {};
  const polygon = room.floor_polygon || [];
  if (polygon.length < 3) return;

  const shape = new THREE.Shape();
  polygon.forEach(([x, y], index) => {
    if (index === 0) shape.moveTo(x, y);
    else shape.lineTo(x, y);
  });
  shape.closePath();

  const floorGeometry = new THREE.ShapeGeometry(shape);
  floorGeometry.rotateX(Math.PI / 2);
  const floorMaterial = new THREE.MeshBasicMaterial({
    color: 0xdde4d7,
    transparent: true,
    opacity: 0.72,
    side: THREE.DoubleSide,
  });
  sceneGroup.add(new THREE.Mesh(floorGeometry, floorMaterial));

  const floorZ = Number(room.floor_z || 0);
  const wallHeight = Number(room.wall_height || 0);
  const bottom = polygon.map(([x, y]) => new THREE.Vector3(x, floorZ, y));
  const top = polygon.map(([x, y]) => new THREE.Vector3(x, wallHeight, y));

  sceneGroup.add(makeLoop(bottom, 0x46534b));
  if (wallHeight > 0) {
    sceneGroup.add(makeLoop(top, 0x8d9890));
    polygon.forEach(([x, y]) => {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x, floorZ, y),
        new THREE.Vector3(x, wallHeight, y),
      ]);
      sceneGroup.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0xb5bcb4 })));
    });
  }
}

function drawObjects(data) {
  const objects = data.objects || [];
  objects.forEach((object) => {
    const size = object.size || [1, 1, 1];
    const center = object.center || [0, 0, 0];
    const invalid = object.validity_status === "invalid";
    const highlighted = highlightedObjectIds.has(object.object_id);
    const geometry = new THREE.BoxGeometry(size[0], size[2], size[1]);
    const material = new THREE.MeshStandardMaterial({
      color: highlighted ? 0xf2c94c : invalid ? 0xd6604d : 0x4f8f75,
      roughness: 0.72,
      metalness: 0.05,
      transparent: true,
      opacity: highlighted ? 0.98 : invalid ? 0.88 : 0.78,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(project(center));
    mesh.rotation.y = -THREE.MathUtils.degToRad(Number(object.yaw || 0));
    sceneGroup.add(mesh);

    const edges = new THREE.EdgesGeometry(geometry);
    const line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: highlighted ? 0x5b3d00 : invalid ? 0x7f1d1d : 0x1f332b })
    );
    line.position.copy(mesh.position);
    line.rotation.copy(mesh.rotation);
    sceneGroup.add(line);

    const label = makeLabel(object.object_id || object.category || "object", invalid, highlighted);
    label.position.set(center[0], center[2] + size[2] / 2 + 0.16, center[1]);
    sceneGroup.add(label);
  });
}

function drawRelations(data) {
  const objectsById = new Map((data.objects || []).map((object) => [object.object_id, object]));
  (data.relations || []).forEach((relation) => {
    const source = objectsById.get(relation.source);
    const target = objectsById.get(relation.target);
    if (!source || !target) return;
    const start = project(source.center || [0, 0, 0]);
    const end = project(target.center || [0, 0, 0]);
    const color = relation.hard ? 0x34495e : 0x6f7f86;
    const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    sceneGroup.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color })));

    const direction = new THREE.Vector3().subVectors(end, start).normalize();
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(0.08, 0.22, 16),
      new THREE.MeshBasicMaterial({ color })
    );
    cone.position.copy(end.clone().add(direction.multiplyScalar(-0.12)));
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    sceneGroup.add(cone);
  });
}

function updatePanel(data, rootData, history) {
  const summary = data.summary || {};
  summaryEl.innerHTML = "";
  [
    ["iteration", data.iteration],
    ["overall_valid", data.overall_valid],
    ["schema_valid", summary.schema_valid],
    ["physical_valid", summary.physical_valid],
    ["spatial_relation_valid", summary.spatial_relation_valid],
  ].forEach(([key, value]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = String(value ?? "n/a");
    summaryEl.append(dt, dd);
  });

  updateWorkflow(data, rootData.workflow || {});
  updateFeedback(rootData.feedback || {});
  updateMetrics(rootData.metrics || {});
  updateViewArtifacts(rootData.view_artifacts || {});
  updateWorkflowTrace(rootData.workflow_trace_path);

  violationsEl.innerHTML = "";
  const violations = data.violations || [];
  if (violations.length === 0) {
    violationsEl.innerHTML = '<div class="item valid">No violations</div>';
  } else {
    violations.forEach((violation) => {
      const div = document.createElement("div");
      div.className = "item invalid clickable";
      div.innerHTML = `<strong>${escapeHtml(violation.category)} / ${escapeHtml(
        violation.type
      )}</strong><br><span class="muted mono">${escapeHtml((violation.objects || []).join(", "))}</span><br>${escapeHtml(
        violation.message || ""
      )}`;
      div.addEventListener("click", () => {
        highlightedObjectIds = new Set(violation.objects || []);
        renderActiveScene();
        updateArtifactPreview(getArtifacts()[selectedArtifactIndex]);
      });
      violationsEl.appendChild(div);
    });
  }

  relationsEl.innerHTML = "";
  const relations = data.relations || [];
  if (relations.length === 0) {
    relationsEl.innerHTML = '<div class="item muted">No relations</div>';
  } else {
    relations.forEach((relation) => {
      const div = document.createElement("div");
      div.className = "item";
      div.innerHTML = `<strong>${escapeHtml(relation.type)}</strong><br><span class="mono">${escapeHtml(
        relation.source || ""
      )} -> ${escapeHtml(relation.target || "")}</span>`;
      relationsEl.appendChild(div);
    });
  }

  historyEl.innerHTML = "";
  if (!Array.isArray(history) || history.length === 0) {
    historyEl.innerHTML = '<div class="item muted">No history</div>';
  } else {
    history.forEach((item) => {
      const div = document.createElement("div");
      div.className = `item ${item.overall_valid ? "valid" : "invalid"}`;
      div.innerHTML = `<strong>iteration ${escapeHtml(item.iteration)}</strong><br>
        overall_valid: ${escapeHtml(item.overall_valid)}<br>
        <span class="mono">${escapeHtml(item.layout_path || "")}</span>`;
      historyEl.appendChild(div);
    });
  }
}

function updateWorkflow(data, workflow) {
  workflowEl.innerHTML = "";
  const artifacts = workflow.artifacts || [];
  if (artifacts.length === 0) {
    workflowEl.innerHTML = '<div class="item muted">No workflow metadata</div>';
    return;
  }

  artifacts.forEach((artifact, index) => {
    const div = document.createElement("button");
    const isCurrentIteration = artifact.iteration === data.iteration;
    const isGlobalStep = artifact.step === "input" || artifact.step === "metrics" || artifact.step === "visualize";
    div.type = "button";
    div.className = `workflow-step ${isCurrentIteration ? "active" : ""} ${
      index === selectedArtifactIndex ? "selected" : ""
    }`;
    div.innerHTML = `
      <div class="step-index">${index + 1}</div>
      <div class="step-body">
        <strong>${escapeHtml(artifact.label || artifact.step)}</strong>
        <span>${escapeHtml(artifact.step || "")}${artifact.status ? ` / ${escapeHtml(artifact.status)}` : ""}</span>
        <code>${escapeHtml(artifact.path || "")}</code>
      </div>
    `;
    if (isGlobalStep && data.iteration !== undefined) {
      div.classList.add("global");
    }
    div.addEventListener("click", () => {
      stopReplay();
      selectArtifact(index, { syncScene: true });
    });
    workflowEl.appendChild(div);
  });
}

function updateArtifactPreview(artifact) {
  if (!artifact) {
    artifactTitleEl.textContent = "Select a workflow step";
    artifactPreviewEl.textContent = "{}";
    return;
  }
  artifactTitleEl.textContent = `${artifact.label || artifact.step} - ${artifact.path || "embedded artifact"}`;
  if (artifact.data !== undefined) {
    artifactPreviewEl.textContent = JSON.stringify(artifact.data, null, 2);
    return;
  }
  if (!artifact.path) {
    artifactPreviewEl.textContent = "{}";
    return;
  }
  artifactPreviewEl.textContent = "Loading...";
  fetch(`./${artifact.path}`, { cache: "no-store" })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Unable to load ${artifact.path}: ${response.status}`);
      }
      return response.json();
    })
    .then((json) => {
      artifactPreviewEl.textContent = JSON.stringify(json, null, 2);
    })
    .catch((error) => {
      artifactPreviewEl.textContent = error.message;
    });
}

function replayWorkflow() {
  const artifacts = getArtifacts();
  if (!artifacts.length) return;
  let index = 0;
  replayButton.textContent = "Stop Replay";
  selectArtifact(index, { syncScene: true });
  replayTimer = window.setInterval(() => {
    index += 1;
    if (index >= artifacts.length) {
      stopReplay();
      return;
    }
    selectArtifact(index, { syncScene: true });
  }, 1200);
}

function stopReplay() {
  if (!replayTimer) return;
  window.clearInterval(replayTimer);
  replayTimer = null;
  replayButton.textContent = "Replay Workflow";
}

function updateFeedback(feedback) {
  feedbackEl.innerHTML = "";
  if (!feedback || Object.keys(feedback).length === 0) {
    feedbackEl.innerHTML = '<div class="item muted">No feedback generated</div>';
    return;
  }

  const targets = (feedback.repair_targets || []).join(", ") || "none";
  const locked = (feedback.locked_objects || []).join(", ") || "none";
  const overview = document.createElement("div");
  overview.className = "item";
  overview.innerHTML = `
    <strong>iteration ${escapeHtml(feedback.iteration)}</strong><br>
    repair_targets: <span class="mono">${escapeHtml(targets)}</span><br>
    locked_objects: <span class="mono">${escapeHtml(locked)}</span><br>
    <span class="muted">${escapeHtml(feedback.instruction || "")}</span>
  `;
  feedbackEl.appendChild(overview);

  (feedback.violations || []).forEach((violation) => {
    const div = document.createElement("div");
    div.className = "item invalid clickable";
    div.innerHTML = `<strong>${escapeHtml(violation.category)} / ${escapeHtml(
      violation.type
    )}</strong><br><span class="mono">${escapeHtml((violation.objects || []).join(", "))}</span><br>${escapeHtml(
      violation.message || ""
    )}`;
    div.addEventListener("click", () => {
      const iteration = Number.isInteger(feedback.iteration) ? feedback.iteration : 0;
      activeScene = findSceneByIteration(iteration);
      const scenes = getScenes(viewerData);
      const sceneIndex = scenes.findIndex((item) => item.iteration === iteration);
      if (sceneIndex >= 0) {
        iterationSelect.value = String(sceneIndex);
      }
      highlightedObjectIds = new Set(violation.objects || []);
      selectedArtifactIndex = findArtifactByStep("feedback", iteration) ?? selectedArtifactIndex;
      renderActiveScene();
      updateArtifactPreview(getArtifacts()[selectedArtifactIndex]);
    });
    feedbackEl.appendChild(div);
  });
}

function updateMetrics(metrics) {
  metricsEl.innerHTML = "";
  const entries = Object.entries(metrics || {});
  if (entries.length === 0) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = "metrics";
    dd.textContent = "n/a";
    metricsEl.append(dt, dd);
    return;
  }
  entries.forEach(([key, value]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = typeof value === "number" ? String(Number(value.toFixed?.(3) ?? value)) : String(value);
    metricsEl.append(dt, dd);
  });
}

function updateViewArtifacts(viewArtifacts) {
  roomViewsEl.innerHTML = "";
  pairViewsEl.innerHTML = "";

  const room = viewArtifacts.room || [];
  if (!room.length) {
    roomViewsEl.innerHTML = '<div class="item muted">No room views</div>';
  } else {
    roomViewsEl.appendChild(makeViewGrid(room));
  }

  const pairItems = [];
  for (const section of [viewArtifacts.relations || [], viewArtifacts.attachments || []]) {
    section.forEach((item) => {
      (item.view_artifacts || []).forEach((artifact) => {
        if (artifact.path && artifact.path.endsWith(".png")) {
          pairItems.push({ ...artifact, id: `${item.id || "pair"} / ${artifact.id}` });
        }
      });
    });
  }
  if (!pairItems.length) {
    pairViewsEl.innerHTML = '<div class="item muted">No pair views</div>';
  } else {
    pairViewsEl.appendChild(makeViewGrid(pairItems));
  }
}

function makeViewGrid(artifacts) {
  const grid = document.createElement("div");
  grid.className = "view-grid";
  artifacts
    .filter((artifact) => artifact.path && artifact.path.endsWith(".png"))
    .forEach((artifact) => {
      const cell = document.createElement("a");
      cell.className = "view-thumb";
      cell.href = artifact.path;
      cell.target = "_blank";
      cell.rel = "noreferrer";
      cell.innerHTML = `<img src="${escapeHtml(artifact.path)}" alt="${escapeHtml(
        artifact.id || "view"
      )}" /><span>${escapeHtml(artifact.id || artifact.path)}</span>`;
      grid.appendChild(cell);
    });
  return grid;
}

function updateWorkflowTrace(tracePath) {
  workflowTraceEl.innerHTML = "";
  if (!tracePath) {
    workflowTraceEl.innerHTML = '<div class="item muted">No workflow trace</div>';
    return;
  }
  const div = document.createElement("div");
  div.className = "item";
  div.innerHTML = `<strong>Trace</strong><br><span class="mono">${escapeHtml(tracePath)}</span>`;
  div.addEventListener("click", () => {
    updateArtifactPreview({ label: "Workflow Trace", path: tracePath });
  });
  workflowTraceEl.appendChild(div);
}

function frameScene(data) {
  const polygon = data.room?.floor_polygon || [];
  if (polygon.length === 0) return;
  const xs = polygon.map((point) => point[0]);
  const ys = polygon.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const center = new THREE.Vector3((minX + maxX) / 2, 0.6, (minY + maxY) / 2);
  const span = Math.max(maxX - minX, maxY - minY, 4);
  controls.target.copy(center);
  camera.position.set(center.x + span * 0.75, span * 0.95, center.z + span * 0.9);
  camera.near = 0.01;
  camera.far = Math.max(100, span * 20);
  camera.updateProjectionMatrix();
  controls.update();
}

function makeLoop(points, color) {
  const closed = [...points, points[0]];
  const geometry = new THREE.BufferGeometry().setFromPoints(closed);
  return new THREE.Line(geometry, new THREE.LineBasicMaterial({ color }));
}

function makeLabel(text, invalid, highlighted = false) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.width = 256;
  canvas.height = 64;
  context.fillStyle = highlighted
    ? "rgba(91, 61, 0, 0.94)"
    : invalid
      ? "rgba(126, 30, 30, 0.92)"
      : "rgba(22, 48, 40, 0.9)";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "white";
  context.font = "24px sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.9, 0.225, 1);
  return sprite;
}

function project(center) {
  return new THREE.Vector3(Number(center[0] || 0), Number(center[2] || 0), Number(center[1] || 0));
}

function clearGroup(group) {
  while (group.children.length > 0) {
    const child = group.children.pop();
    child.traverse?.((node) => {
      node.geometry?.dispose?.();
      if (Array.isArray(node.material)) {
        node.material.forEach((material) => material.dispose?.());
      } else {
        node.material?.dispose?.();
      }
    });
  }
}

function resize() {
  const width = viewport.clientWidth || 1;
  const height = viewport.clientHeight || 1;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
