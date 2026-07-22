import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/controls/OrbitControls.js";

const viewport = document.getElementById("viewport");
const titleEl = document.getElementById("task-title");
const iterationSelect = document.getElementById("iteration-select");
const beforeButton = document.getElementById("before-button");
const afterButton = document.getElementById("after-button");
const compareButton = document.getElementById("compare-button");
const replayButton = document.getElementById("replay-button");
const groupColorButton = document.getElementById("group-color-button");
const edgeOverlayButton = document.getElementById("edge-overlay-button");
const replayPrevButton = document.getElementById("replay-prev-button");
const replayNextButton = document.getElementById("replay-next-button");
const replayResetButton = document.getElementById("replay-reset-button");
const appEl = document.getElementById("app");
const summaryEl = document.getElementById("summary");
const workflowEl = document.getElementById("workflow");
const artifactTitleEl = document.getElementById("artifact-title");
const artifactToggleEl = document.getElementById("artifact-toggle");
const artifactLoadEl = document.getElementById("artifact-load");
const artifactCopyEl = document.getElementById("artifact-copy");
const artifactResizeHandleEl = document.getElementById("artifact-resize-handle");
const artifactPreviewEl = document.getElementById("artifact-preview");
const violationsEl = document.getElementById("violations");
const feedbackEl = document.getElementById("feedback");
const metricsEl = document.getElementById("metrics");
const globalViewsEl = document.getElementById("global-views");
const groupEvidenceEl = document.getElementById("group-evidence");
const skippedObjectsEl = document.getElementById("skipped-objects");
const judgeArtifactsEl = document.getElementById("judge-artifacts");
const roomViewsEl = document.getElementById("room-views");
const pairViewsEl = document.getElementById("pair-views");
const workflowTraceEl = document.getElementById("workflow-trace");
const relationsEl = document.getElementById("relations");
const historyEl = document.getElementById("history");

const DRAWER_STORAGE_KEY = "layoutBenchmarkArtifactDrawer";
const DEFAULT_DRAWER_HEIGHT = 260;
const COLLAPSED_DRAWER_HEIGHT = 38;
const MIN_DRAWER_HEIGHT = 80;
const MAX_DRAWER_HEIGHT_RATIO = 0.65;
let maxPreviewChars = 20000;
const INITIAL_CAMERA_DISTANCE_SCALE = 1.2;
const INITIAL_SCREEN_OFFSET_X = -0.08;
const INITIAL_SCREEN_OFFSET_Y = 0.08;
const RELATION_EDGE_RADIUS = 0.018;
const FORMATION_EDGE_RADIUS = 0.012;
const RELATION_ARROW_RADIUS = 0.096;
const RELATION_ARROW_HEIGHT = 0.264;
const RELATION_EDGE_COLOR = 0x111111;

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
let groupEvidenceFilter = "all";
let artifactDrawerCollapsed = false;
let currentArtifact = null;
let colorByGroup = false;
let showRelationEdges = false;
let compareMode = false;
let selectedGroupId = null;
let objectMeshes = [];
let currentDiffObjectIds = new Set();
let cameraFramedOnce = false;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

fetch("./viewer_scene.json", { cache: "no-store" })
  .then((response) => {
    if (!response.ok) {
      throw new Error(`Unable to load viewer_scene.json: ${response.status}`);
    }
    return response.json();
  })
  .then((data) => {
    viewerData = data;
    maxPreviewChars = Number(data.viewer_options?.json_preview?.truncate_chars || maxPreviewChars);
    colorByGroup = Boolean(data.viewer_options?.group_coloring?.enabled_by_default);
    showRelationEdges = Boolean(data.viewer_options?.overlays?.show_relation_edges_by_default);
    setDrawerCollapsed(Boolean(data.viewer_options?.json_preview?.hidden_by_default), { persist: false });
    syncVisualButtons();
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
  selectedGroupId = null;
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

compareButton.addEventListener("click", () => {
  stopReplay();
  compareMode = !compareMode;
  compareButton.classList.toggle("active", compareMode);
  compareButton.setAttribute("aria-pressed", String(compareMode));
  renderActiveScene();
});

replayButton.addEventListener("click", () => {
  if (replayTimer) {
    stopReplay();
    return;
  }
  replayWorkflow();
});

groupColorButton.addEventListener("click", () => {
  colorByGroup = !colorByGroup;
  syncVisualButtons();
  renderActiveScene();
});

edgeOverlayButton.addEventListener("click", () => {
  showRelationEdges = !showRelationEdges;
  syncVisualButtons();
  renderActiveScene();
});

replayPrevButton.addEventListener("click", () => {
  stopReplay();
  selectArtifact(selectedArtifactIndex - 1, { syncScene: true });
});

replayNextButton.addEventListener("click", () => {
  stopReplay();
  selectArtifact(selectedArtifactIndex + 1, { syncScene: true });
});

replayResetButton.addEventListener("click", () => {
  stopReplay();
  compareMode = false;
  compareButton.classList.remove("active");
  compareButton.setAttribute("aria-pressed", "false");
  selectedGroupId = null;
  highlightedObjectIds = new Set();
  selectArtifact(0, { syncScene: true, forceFrame: true });
});

artifactLoadEl.addEventListener("click", () => loadArtifactPreview(currentArtifact));
artifactCopyEl.addEventListener("click", () => copyArtifactPath(currentArtifact));
renderer.domElement.addEventListener("pointerdown", pickObject);

initArtifactDrawer();
window.addEventListener("resize", resize);
resize();
animate();

function initArtifactDrawer() {
  const stored = readDrawerState();
  artifactDrawerCollapsed = stored.collapsed;
  setDrawerHeight(stored.height || DEFAULT_DRAWER_HEIGHT, { persist: false });
  setDrawerCollapsed(artifactDrawerCollapsed, { persist: false });

  artifactToggleEl.addEventListener("click", () => {
    setDrawerCollapsed(!artifactDrawerCollapsed);
  });

  artifactResizeHandleEl.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    setDrawerCollapsed(false);
    artifactResizeHandleEl.setPointerCapture?.(event.pointerId);
    const move = (moveEvent) => {
      const maxHeight = Math.max(MIN_DRAWER_HEIGHT, Math.floor(window.innerHeight * MAX_DRAWER_HEIGHT_RATIO));
      const nextHeight = clamp(window.innerHeight - moveEvent.clientY, MIN_DRAWER_HEIGHT, maxHeight);
      setDrawerHeight(nextHeight);
    };
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
  });
}

function readDrawerState() {
  try {
    return JSON.parse(window.localStorage.getItem(DRAWER_STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

function writeDrawerState(patch) {
  const state = { ...readDrawerState(), ...patch };
  window.localStorage.setItem(DRAWER_STORAGE_KEY, JSON.stringify(state));
}

function setDrawerCollapsed(collapsed, options = {}) {
  artifactDrawerCollapsed = collapsed;
  appEl.classList.toggle("drawer-collapsed", collapsed);
  artifactToggleEl.textContent = collapsed ? "Expand" : "Collapse";
  artifactToggleEl.setAttribute("aria-expanded", String(!collapsed));
  if (collapsed) {
    appEl.style.setProperty("--artifact-drawer-height", `${COLLAPSED_DRAWER_HEIGHT}px`);
  } else {
    setDrawerHeight(readDrawerState().height || DEFAULT_DRAWER_HEIGHT, { persist: false });
  }
  if (options.persist !== false) {
    writeDrawerState({ collapsed });
  }
  window.requestAnimationFrame(resize);
}

function setDrawerHeight(height, options = {}) {
  const maxHeight = Math.max(MIN_DRAWER_HEIGHT, Math.floor(window.innerHeight * MAX_DRAWER_HEIGHT_RATIO));
  const nextHeight = clamp(Number(height) || DEFAULT_DRAWER_HEIGHT, MIN_DRAWER_HEIGHT, maxHeight);
  appEl.style.setProperty("--artifact-drawer-height", `${nextHeight}px`);
  if (options.persist !== false) {
    writeDrawerState({ height: nextHeight });
  }
  window.requestAnimationFrame(resize);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

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
  const hasRepair = scenes.some((item) => Number(item.iteration) > 0);
  afterButton.disabled = !hasRepair;
  compareButton.disabled = !hasRepair;
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
  selectedGroupId = null;
  renderActiveScene({ forceFrame: Boolean(options.forceFrame) });
  updateArtifactPreview(artifact);
}

function renderActiveScene(options = {}) {
  if (!activeScene) return;
  clearGroup(sceneGroup);
  objectMeshes = [];
  titleEl.textContent = activeScene.task_id || "viewer_scene.json";
  currentDiffObjectIds = compareMode ? new Set(activeScene.diff_from_initial?.changed_object_ids || activeScene.diff_from_previous?.changed_object_ids || []) : new Set();

  drawRoom(activeScene);
  drawObjects(activeScene);
  drawRelations(activeScene);
  updatePanel(activeScene, viewerData || {}, viewerData?.history || activeScene.history || []);
  if (options.forceFrame || !cameraFramedOnce) {
    frameScene(activeScene);
    cameraFramedOnce = true;
  } else {
    camera.lookAt(controls.target);
    controls.update();
  }
}

function drawRoom(data) {
  const room = data.room || {};
  const options = data.viewer_options || viewerData?.viewer_options || {};
  const overlays = options.overlays || {};
  const regions = floorPlanRegions(room);
  const polygon = room.floor_polygon || room.floor_plan?.aggregate_boundary || [];
  const floorZ = Number(room.floor_z || 0);
  const wallHeight = Number(room.wall_height || 0);
  const bounds = roomBoundsFromRoom(room);

  if (regions.length && overlays.show_floor_plan_regions !== false) {
    regions.forEach((region, index) => drawFloorRegion(region, index, floorZ));
  }

  if (overlays.show_floor_grid !== false) {
    drawFloorGrid(bounds, floorZ);
  }
  if (overlays.show_axes !== false) {
    drawAxes(bounds, floorZ);
  }

  if (!Array.isArray(polygon) || polygon.length < 3) return;

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
    opacity: regions.length ? 0.16 : 0.72,
    side: THREE.DoubleSide,
  });
  if (overlays.show_room_proxy !== false) {
    sceneGroup.add(new THREE.Mesh(floorGeometry, floorMaterial));
  }

  const bottom = polygon.map(([x, y]) => new THREE.Vector3(x, floorZ, y));
  const top = polygon.map(([x, y]) => new THREE.Vector3(x, wallHeight, y));

  if (overlays.show_room_proxy !== false) {
    sceneGroup.add(makeLoop(bottom, 0x46534b));
  }
  if (overlays.show_room_proxy !== false && wallHeight > 0) {
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
  const groupById = new Map((data.group_evidence || []).map((group) => [group.group_id, group]));
  objects.forEach((object) => {
    const size = object.size || [1, 1, 1];
    const center = object.center || [0, 0, 0];
    const invalid = object.validity_status === "invalid";
    const highlighted = highlightedObjectIds.has(object.object_id);
    const changed = currentDiffObjectIds.has(object.object_id);
    const group = groupById.get(object.group_id);
    const baseColor = colorByGroup && object.group_color_key ? groupColor(object.group_color_key) : 0x4f8f75;
    const geometry = new THREE.BoxGeometry(size[0], size[2], size[1]);
    const material = new THREE.MeshStandardMaterial({
      color: highlighted ? 0xf2c94c : changed ? 0x8f5fbf : invalid ? 0xd6604d : baseColor,
      roughness: 0.72,
      metalness: 0.05,
      transparent: true,
      opacity: highlighted ? 0.98 : invalid ? 0.88 : 0.78,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(project(center));
    mesh.rotation.y = -THREE.MathUtils.degToRad(Number(object.yaw || 0));
    mesh.userData = { objectId: object.object_id, groupId: object.group_id };
    sceneGroup.add(mesh);
    objectMeshes.push(mesh);

    const edges = new THREE.EdgesGeometry(geometry);
    const outlineColor = highlighted
      ? 0x5b3d00
      : changed
        ? 0x5d2f91
        : object.sent_to_judge === true
          ? 0x0f6a4d
          : object.sent_to_judge === false
            ? 0x8a4f18
            : invalid
              ? 0x7f1d1d
              : 0x1f332b;
    const line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: outlineColor })
    );
    line.position.copy(mesh.position);
    line.rotation.copy(mesh.rotation);
    sceneGroup.add(line);

    const label = makeLabel(object.object_id || object.category || "object", invalid, highlighted);
    label.position.set(center[0], center[2] + size[2] / 2 + 0.16, center[1]);
    sceneGroup.add(label);
    const statusLabel = judgeStatusLabel(group);
    if (statusLabel && viewerData?.viewer_options?.overlays?.show_judge_status_markers !== false) {
      const badge = makeStatusBadge(statusLabel);
      badge.position.set(center[0], center[2] + size[2] / 2 + 0.42, center[1]);
      sceneGroup.add(badge);
    }
  });
}

function drawRelations(data) {
  if (!showRelationEdges) return;
  const objectsById = new Map((data.objects || []).map((object) => [object.object_id, object]));
  (data.relations || []).forEach((relation) => {
    const source = objectsById.get(relation.source);
    const target = objectsById.get(relation.target);
    if (!source || !target) return;
    const start = project(source.center || [0, 0, 0]);
    const end = project(target.center || [0, 0, 0]);
    const color = RELATION_EDGE_COLOR;
    const edgeMesh = makeThickLine(start, end, color, RELATION_EDGE_RADIUS);
    if (!edgeMesh) return;
    sceneGroup.add(edgeMesh);

    const direction = new THREE.Vector3().subVectors(end, start).normalize();
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(RELATION_ARROW_RADIUS, RELATION_ARROW_HEIGHT, 16),
      new THREE.MeshBasicMaterial({ color })
    );
    cone.position.copy(end.clone().add(direction.multiplyScalar(-RELATION_ARROW_HEIGHT * 0.55)));
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    sceneGroup.add(cone);
  });
  (data.group_evidence || []).forEach((group) => {
    (group.formation_edges || []).forEach((edge) => {
      const source = objectsById.get(edge.source);
      const target = objectsById.get(edge.target);
      if (!source || !target) return;
      const color = RELATION_EDGE_COLOR;
      const edgeMesh = makeThickLine(
        project(source.center || [0, 0, 0]),
        project(target.center || [0, 0, 0]),
        color,
        FORMATION_EDGE_RADIUS,
        0.78
      );
      if (edgeMesh) sceneGroup.add(edgeMesh);
    });
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
    ["floor_plan_source", data.room?.floor_plan?.source],
    ["floor_plan_regions", data.room?.floor_plan?.region_count],
    ["floor_plan_labels", floorPlanRegions(data.room || {}).map((region) => region.label || region.name || region.id).join(", ")],
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
  updateGlobalViews(data, rootData.view_artifacts || {});
  updateGroupEvidence(data);
  updateSkippedObjects(data);
  updateJudgeArtifacts(data);
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
  currentArtifact = artifact || null;
  if (!artifact) {
    artifactTitleEl.textContent = "Select a workflow step";
    artifactPreviewEl.textContent = "No artifact selected.";
    artifactLoadEl.disabled = true;
    artifactCopyEl.disabled = true;
    return;
  }
  artifactTitleEl.textContent = `${artifact.label || artifact.step} - ${artifact.path || "embedded artifact"}`;
  artifactLoadEl.disabled = !artifact.path && artifact.data === undefined;
  artifactCopyEl.disabled = !artifact.path;
  const metadata = {
    step: artifact.step,
    label: artifact.label,
    status: artifact.status,
    iteration: artifact.iteration,
    path: artifact.path,
  };
  artifactPreviewEl.textContent = `Artifact metadata loaded.\nClick "Load artifact" to fetch the JSON/text body lazily.\n\n${formatPreviewPayload(metadata)}`;
}

function loadArtifactPreview(artifact) {
  if (!artifact) {
    artifactPreviewEl.textContent = "No artifact selected.";
    return;
  }
  if (artifact.data !== undefined) {
    artifactPreviewEl.textContent = formatPreviewPayload(artifact.data);
    return;
  }
  if (!artifact.path) {
    artifactPreviewEl.textContent = "No artifact path.";
    return;
  }
  artifactPreviewEl.textContent = "Loading...";
  fetch(`./${artifact.path}`, { cache: "no-store" })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Unable to load ${artifact.path}: ${response.status}`);
      }
      return artifact.path.endsWith(".json") ? response.json() : response.text();
    })
    .then((payload) => {
      artifactPreviewEl.textContent = formatPreviewPayload(payload);
    })
    .catch((error) => {
      artifactPreviewEl.textContent = error.message;
    });
}

function formatPreviewPayload(payload) {
  const text = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
  if (text.length <= maxPreviewChars) {
    return text;
  }
  return `${text.slice(0, maxPreviewChars)}\n\n... preview truncated at ${maxPreviewChars} characters ...`;
}

function replayWorkflow() {
  const artifacts = getArtifacts();
  if (!artifacts.length) return;
  let index = selectedArtifactIndex || 0;
  const duration = Number(viewerData?.viewer_options?.replay?.step_duration_ms || 1200);
  replayButton.textContent = "Stop Replay";
  selectArtifact(index, { syncScene: true });
  replayTimer = window.setInterval(() => {
    index += 1;
    if (index >= artifacts.length) {
      stopReplay();
      return;
    }
    selectArtifact(index, { syncScene: true });
  }, duration);
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

function updateGlobalViews(data, viewArtifacts) {
  globalViewsEl.innerHTML = "";
  const views = data.global_views || viewArtifacts.global || viewArtifacts.room || [];
  if (!views.length) {
    globalViewsEl.innerHTML = '<div class="item muted">No global view</div>';
    return;
  }
  globalViewsEl.appendChild(makeViewGrid(views));
}

function updateGroupEvidence(data) {
  groupEvidenceEl.innerHTML = "";
  const groups = data.group_evidence || [];
  if (!groups.length) {
    groupEvidenceEl.innerHTML = '<div class="item muted">No group evidence</div>';
    return;
  }

  const controls = document.createElement("div");
  controls.className = "group-filter-row";
  [
    ["all", "All Groups"],
    ["selected", "Selected for Judge"],
    ["omitted", "Omitted from Judge"],
  ].forEach(([value, label]) => {
    const control = document.createElement("button");
    control.type = "button";
    control.className = `group-filter ${groupEvidenceFilter === value ? "active" : ""}`;
    control.textContent = label;
    control.addEventListener("click", () => {
      groupEvidenceFilter = value;
      updateGroupEvidence(data);
    });
    controls.appendChild(control);
  });
  groupEvidenceEl.appendChild(controls);

  const filteredGroups = groups.filter((group) => {
    if (groupEvidenceFilter === "selected") return group.sent_to_judge === true;
    if (groupEvidenceFilter === "omitted") return group.sent_to_judge === false;
    return true;
  });
  if (!filteredGroups.length) {
    groupEvidenceEl.insertAdjacentHTML("beforeend", '<div class="item muted">No groups in this filter</div>');
    return;
  }

  filteredGroups.forEach((group) => {
    const objectIds = group.object_ids || [];
    const judgeState = group.sent_to_judge === true ? "selected" : group.sent_to_judge === false ? "omitted" : "unbudgeted";
    const button = document.createElement("button");
    const isActive = selectedGroupId === group.group_id || objectIds.some((objectId) => highlightedObjectIds.has(objectId));
    button.type = "button";
    button.className = `group-button ${isActive ? "active" : ""}`;
    button.innerHTML = `
      <strong>${escapeHtml(group.group_label || group.group_id || "group")}</strong>
      <span>${escapeHtml(judgeState)} / score ${escapeHtml(String(group.selection_score ?? "n/a"))}</span>
      <span class="mono">${escapeHtml(objectIds.join(", ") || "no objects")}</span>
    `;
    button.addEventListener("click", () => {
      selectedGroupId = group.group_id;
      highlightedObjectIds = new Set(objectIds);
      renderActiveScene();
      updateArtifactPreview({ label: `Group Evidence ${group.group_id || ""}`, data: group });
    });
    groupEvidenceEl.appendChild(button);

    if (!isActive) return;
    const detail = document.createElement("div");
    detail.className = "item";
    detail.innerHTML = `
      <strong>Judge Selection</strong><br>
      ${escapeHtml(judgeState)} / score ${escapeHtml(String(group.selection_score ?? "n/a"))}<br>
      <span class="muted">${escapeHtml((group.selection_reasons || []).join(", ") || "no selection reasons")}</span><br><br>
      <strong>Formation</strong><br>
      ${formatFormationEdges(group.formation_edges || group.edge_reasons || [])}
    `;
    const viewItems = groupViewsToList(group.views || {});
    if (viewItems.length) {
      detail.appendChild(makeViewGrid(viewItems));
    }
    const diagnostics = {
      diagnostics: group.diagnostics || {},
      view_flags: group.view_flags || [],
      relations: group.relations || [],
      skipped_objects: group.skipped_objects || [],
    };
    const pre = document.createElement("pre");
    pre.className = "diagnostics-preview";
    pre.textContent = JSON.stringify(diagnostics, null, 2);
    detail.appendChild(pre);
    groupEvidenceEl.appendChild(detail);
  });
}

function updateSkippedObjects(data) {
  skippedObjectsEl.innerHTML = "";
  const skipped = data.skipped_objects || [];
  if (!skipped.length) {
    skippedObjectsEl.innerHTML = '<div class="item valid">No skipped objects</div>';
    return;
  }
  skipped.forEach((item) => {
    const div = document.createElement("div");
    div.className = "item invalid clickable";
    div.innerHTML = `<strong>${escapeHtml(item.object_id || `object ${item.object_index ?? ""}`)}</strong><br>
      <span class="muted">${escapeHtml(item.reason || "")}</span>`;
    div.addEventListener("click", () => updateArtifactPreview({ label: "Skipped Object", data: item }));
    skippedObjectsEl.appendChild(div);
  });
}

function updateJudgeArtifacts(data) {
  judgeArtifactsEl.innerHTML = "";
  const artifacts = data.vlm_judge_artifacts || {};
  const entries = Object.entries(artifacts).filter(([, path]) => path);
  if (!entries.length) {
    judgeArtifactsEl.innerHTML = '<div class="item muted">No judge artifacts</div>';
    return;
  }
  entries.forEach(([key, path]) => {
    const div = document.createElement("div");
    div.className = "item clickable";
    div.innerHTML = `<strong>${escapeHtml(key)}</strong><br><span class="mono">${escapeHtml(path)}</span>`;
    div.addEventListener("click", () => updateArtifactPreview({ label: key, path }));
    judgeArtifactsEl.appendChild(div);
  });
}

function groupViewsToList(views) {
  return ["xy", "yz", "xz"]
    .map((projection) => {
      const artifact = views[projection];
      if (!artifact || !artifact.path) return null;
      return { ...artifact, id: projection };
    })
    .filter(Boolean);
}

function formatFormationEdges(edges) {
  if (!edges.length) return '<span class="muted">No formation edges</span>';
  if (typeof edges[0] === "string") {
    return edges.map((edge) => `<span class="mono">${escapeHtml(edge)}</span>`).join("<br>");
  }
  return edges
    .map(
      (edge) =>
        `<span class="mono">${escapeHtml(edge.source || "")} -> ${escapeHtml(edge.target || "")}</span><br>` +
        `<span class="muted">${escapeHtml(edge.reason || "")} / ${escapeHtml(edge.strength || "")}</span>`
    )
    .join("<br>");
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
      cell.innerHTML = `<img loading="lazy" src="${escapeHtml(artifact.path)}" alt="${escapeHtml(
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

function syncVisualButtons() {
  groupColorButton.classList.toggle("active", colorByGroup);
  groupColorButton.setAttribute("aria-pressed", String(colorByGroup));
  edgeOverlayButton.classList.toggle("active", showRelationEdges);
  edgeOverlayButton.setAttribute("aria-pressed", String(showRelationEdges));
}

function copyArtifactPath(artifact) {
  if (!artifact?.path) return;
  navigator.clipboard?.writeText(artifact.path).then(
    () => {
      artifactCopyEl.textContent = "Copied";
      window.setTimeout(() => (artifactCopyEl.textContent = "Copy path"), 900);
    },
    () => {
      artifactPreviewEl.textContent = artifact.path;
    }
  );
}

function pickObject(event) {
  if (!objectMeshes.length) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(objectMeshes, false)[0];
  if (!hit) return;
  const objectId = hit.object.userData.objectId;
  const groupId = hit.object.userData.groupId;
  selectedGroupId = groupId || null;
  const group = (activeScene?.group_evidence || []).find((item) => item.group_id === groupId);
  highlightedObjectIds = new Set(group?.object_ids?.length ? group.object_ids : [objectId]);
  renderActiveScene();
  const objectRecord = (activeScene?.objects || []).find((item) => item.object_id === objectId);
  updateArtifactPreview({ label: `Object ${objectId}`, data: { object: objectRecord, group } });
}

function roomBounds(polygon) {
  const xs = polygon.map((point) => Number(point[0] || 0));
  const ys = polygon.map((point) => Number(point[1] || 0));
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}

function roomBoundsFromRoom(room) {
  const regionPoints = floorPlanRegions(room).flatMap((region) => region.floor_polygon || []);
  const points = regionPoints.length ? regionPoints : room.floor_polygon || room.floor_plan?.aggregate_boundary || [[0, 0], [4, 0], [4, 4], [0, 4]];
  return roomBounds(points);
}

function floorPlanRegions(room) {
  const regions = room.floor_plan?.regions || [];
  if (!Array.isArray(regions)) return [];
  return regions.filter((region) => Array.isArray(region.floor_polygon) && region.floor_polygon.length >= 3);
}

function drawFloorRegion(region, index, floorZ) {
  const polygon = region.floor_polygon || [];
  const shape = new THREE.Shape();
  polygon.forEach(([x, y], pointIndex) => {
    if (pointIndex === 0) shape.moveTo(x, y);
    else shape.lineTo(x, y);
  });
  shape.closePath();
  const geometry = new THREE.ShapeGeometry(shape);
  geometry.rotateX(Math.PI / 2);
  const color = regionColor(index);
  const material = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.34,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.y = floorZ + 0.004 + index * 0.0006;
  sceneGroup.add(mesh);
  const loop = makeLoop(polygon.map(([x, y]) => new THREE.Vector3(x, floorZ + 0.012, y)), 0x5f6f63);
  sceneGroup.add(loop);

  const center = polygonCenter(polygon);
  const labelText = region.label || region.name || region.id || `region_${index + 1}`;
  const label = makeLabel(labelText, false, false);
  label.position.set(center[0], floorZ + 0.08, center[1]);
  label.scale.set(0.62, 0.155, 1);
  sceneGroup.add(label);
}

function regionColor(index) {
  const palette = [0xdde4d7, 0xd8e1ec, 0xe9dccf, 0xe4d8e9, 0xd7e6e3, 0xe8e2c8, 0xdce8d4];
  return palette[index % palette.length];
}

function polygonCenter(polygon) {
  const xs = polygon.map((point) => Number(point[0] || 0));
  const ys = polygon.map((point) => Number(point[1] || 0));
  return [xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length), ys.reduce((a, b) => a + b, 0) / Math.max(1, ys.length)];
}

function drawFloorGrid(bounds, floorZ) {
  const width = Math.max(1, bounds.maxX - bounds.minX);
  const depth = Math.max(1, bounds.maxY - bounds.minY);
  const size = Math.ceil(Math.max(width, depth));
  const divisions = Math.max(2, Math.min(40, size));
  const grid = new THREE.GridHelper(size, divisions, 0x9aa79d, 0xd4dcd2);
  grid.position.set((bounds.minX + bounds.maxX) / 2, floorZ + 0.002, (bounds.minY + bounds.maxY) / 2);
  sceneGroup.add(grid);
}

function drawAxes(bounds, floorZ) {
  const length = Math.max(1.2, Math.min(3.0, Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) * 0.18));
  const origin = new THREE.Vector3(bounds.minX, floorZ + 0.03, bounds.minY);
  drawAxisLine(origin, new THREE.Vector3(origin.x + length, origin.y, origin.z), 0xc0392b, "x");
  drawAxisLine(origin, new THREE.Vector3(origin.x, origin.y, origin.z + length), 0x2f6fbd, "y");
  drawAxisLine(origin, new THREE.Vector3(origin.x, origin.y + length, origin.z), 0x2f8f5f, "z");
}

function drawAxisLine(start, end, color, label) {
  sceneGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([start, end]), new THREE.LineBasicMaterial({ color })));
  const tag = makeLabel(label, false, true);
  tag.position.copy(end);
  tag.scale.set(0.35, 0.09, 1);
  sceneGroup.add(tag);
}

function groupColor(key) {
  const palette = [0x4f8f75, 0x5b7fb9, 0xb77954, 0x8b6bb1, 0x4f9a9a, 0xb38a3c, 0x7aa85d, 0xb55f76, 0x6f8796];
  return palette[Math.abs(hashString(String(key))) % palette.length];
}

function hashString(value) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) | 0;
  }
  return hash;
}

function judgeStatusLabel(group) {
  if (!group) return "";
  if (group.sent_to_judge === true) return "J";
  if (group.sent_to_judge === false) return "O";
  return "F";
}

function makeStatusBadge(text) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.width = 64;
  canvas.height = 64;
  const color = text === "J" ? "rgba(15, 106, 77, 0.94)" : text === "O" ? "rgba(138, 79, 24, 0.94)" : "rgba(69, 78, 72, 0.86)";
  context.fillStyle = color;
  context.beginPath();
  context.arc(32, 32, 28, 0, Math.PI * 2);
  context.fill();
  context.fillStyle = "white";
  context.font = "bold 30px sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, 32, 34);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
  sprite.scale.set(0.28, 0.28, 1);
  return sprite;
}

function makeThickLine(start, end, color, radius, opacity = 1) {
  const distance = start.distanceTo(end);
  if (!Number.isFinite(distance) || distance <= 1.0e-6) return null;
  const curve = new THREE.CatmullRomCurve3([start, end]);
  const geometry = new THREE.TubeGeometry(curve, 1, radius, 8, false);
  const material = new THREE.MeshBasicMaterial({
    color,
    transparent: opacity < 1,
    opacity,
  });
  return new THREE.Mesh(geometry, material);
}

function frameScene(data) {
  const fallbackBounds = sceneFrameBounds(data);
  if (!fallbackBounds) return;
  const bounds = objectFrameBounds(data.objects || []) || fallbackBounds;
  const center = boundsCenter(bounds);
  const sizeX = Math.max(bounds.maxX - bounds.minX, 1);
  const sizeY = Math.max(bounds.maxY - bounds.minY, 1);
  const sizeZ = Math.max(bounds.maxZ - bounds.minZ, 1);
  const span = Math.max(sizeX, sizeY, sizeZ * 1.35, 4);
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const horizontalFov = 2 * Math.atan(Math.tan(fov / 2) * Math.max(camera.aspect, 0.1));
  const fitHeightDistance = span / (2 * Math.tan(fov / 2));
  const fitWidthDistance = span / (2 * Math.tan(horizontalFov / 2));
  const distance = Math.max(fitHeightDistance, fitWidthDistance) * INITIAL_CAMERA_DISTANCE_SCALE;
  const direction = new THREE.Vector3(0.68, 0.64, 0.72).normalize();
  controls.target.copy(center);
  camera.position.copy(center).addScaledVector(direction, distance);
  camera.lookAt(center);
  recenterCameraToProjectedBounds(bounds);
  camera.near = 0.01;
  camera.far = Math.max(100, distance * 8);
  camera.updateProjectionMatrix();
  controls.update();
}

function recenterCameraToProjectedBounds(bounds) {
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();
  const points = boundsCorners(bounds);
  const projected = points.map((point) => point.clone().project(camera)).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (!projected.length) return;
  const minX = Math.min(...projected.map((point) => point.x));
  const maxX = Math.max(...projected.map((point) => point.x));
  const minY = Math.min(...projected.map((point) => point.y));
  const maxY = Math.max(...projected.map((point) => point.y));
  const offsetX = (minX + maxX) / 2 - INITIAL_SCREEN_OFFSET_X;
  const offsetY = (minY + maxY) / 2 - INITIAL_SCREEN_OFFSET_Y;
  if (Math.abs(offsetX) < 0.02 && Math.abs(offsetY) < 0.02) return;

  const targetDistance = camera.position.distanceTo(controls.target);
  const visibleHeight = 2 * targetDistance * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
  const visibleWidth = visibleHeight * Math.max(camera.aspect, 0.1);
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).normalize();
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1).normalize();
  const shift = right
    .multiplyScalar(offsetX * visibleWidth * 0.5)
    .add(up.multiplyScalar(offsetY * visibleHeight * 0.5));
  camera.position.add(shift);
  controls.target.add(shift);
  camera.lookAt(controls.target);
}

function boundsCorners(bounds) {
  return [
    new THREE.Vector3(bounds.minX, bounds.minZ, bounds.minY),
    new THREE.Vector3(bounds.minX, bounds.minZ, bounds.maxY),
    new THREE.Vector3(bounds.minX, bounds.maxZ, bounds.minY),
    new THREE.Vector3(bounds.minX, bounds.maxZ, bounds.maxY),
    new THREE.Vector3(bounds.maxX, bounds.minZ, bounds.minY),
    new THREE.Vector3(bounds.maxX, bounds.minZ, bounds.maxY),
    new THREE.Vector3(bounds.maxX, bounds.maxZ, bounds.minY),
    new THREE.Vector3(bounds.maxX, bounds.maxZ, bounds.maxY),
  ];
}

function boundsCenter(bounds) {
  return new THREE.Vector3(
    (bounds.minX + bounds.maxX) / 2,
    (bounds.minZ + bounds.maxZ) / 2,
    (bounds.minY + bounds.maxY) / 2
  );
}

function objectFrameBounds(objects) {
  const xs = [];
  const ys = [];
  const zs = [];
  objects.forEach((object) => {
    const center = object.center || [0, 0, 0];
    const size = object.size || [1, 1, 1];
    const x = Number(center[0] || 0);
    const y = Number(center[1] || 0);
    const z = Number(center[2] || 0);
    const halfX = Math.max(Number(size[0] || 1), 0) / 2;
    const halfY = Math.max(Number(size[1] || 1), 0) / 2;
    const halfZ = Math.max(Number(size[2] || 1), 0) / 2;
    xs.push(x - halfX, x + halfX);
    ys.push(y - halfY, y + halfY);
    zs.push(z - halfZ, z + halfZ);
  });
  if (!xs.length || !ys.length || !zs.length) return null;
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
    minZ: Math.min(...zs),
    maxZ: Math.max(...zs),
  };
}

function sceneFrameBounds(data) {
  const room = data.room || {};
  const xs = [];
  const ys = [];
  const floorZ = Number(room.floor_z || 0);
  const wallHeight = Number(room.wall_height || 0);
  const zs = [floorZ];
  if (wallHeight > floorZ) zs.push(wallHeight);

  const regionPoints = floorPlanRegions(room).flatMap((region) => region.floor_polygon || []);
  const polygon = regionPoints.length ? regionPoints : room.floor_polygon || room.floor_plan?.aggregate_boundary || [];
  polygon.forEach((point) => {
    if (!Array.isArray(point) || point.length < 2) return;
    xs.push(Number(point[0] || 0));
    ys.push(Number(point[1] || 0));
  });

  (data.objects || []).forEach((object) => {
    const center = object.center || [0, 0, 0];
    const size = object.size || [1, 1, 1];
    const x = Number(center[0] || 0);
    const y = Number(center[1] || 0);
    const z = Number(center[2] || 0);
    const halfX = Math.max(Number(size[0] || 1), 0) / 2;
    const halfY = Math.max(Number(size[1] || 1), 0) / 2;
    const halfZ = Math.max(Number(size[2] || 1), 0) / 2;
    xs.push(x - halfX, x + halfX);
    ys.push(y - halfY, y + halfY);
    zs.push(z - halfZ, z + halfZ);
  });

  if (!xs.length || !ys.length || !zs.length) return null;
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
    minZ: Math.min(...zs),
    maxZ: Math.max(...zs),
  };
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
