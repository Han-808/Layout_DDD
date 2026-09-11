/*
 * Browser-side game scene probe (game_scene_probe_v1).
 *
 * Evaluated inside a headless page after the game has been stepped to a fixed
 * tick. Walks the three.js scene graph and reports one entry per visible mesh,
 * in the game's own world frame and units. Frame conversion, unit scaling, and
 * OBB fitting all happen on the Python side so that the canonical scene and the
 * collision meshes are derived from the same vertices.
 *
 * Individualization is at the mesh, not at the group. Graph depth is an
 * authoring accident -- one generated implementation writes a flat `addBox()`
 * helper and another nests five levels of Group for the same kind of static
 * level -- so no fixed depth is right for every game, and a rule that reads
 * depth cannot tell when it is wrong. The mesh is the one level that every
 * implementation necessarily has. Grouping decisions that survive are made
 * downstream from geometry, which is authoring-independent.
 *
 * The probe classifies but never discards: meshes that look non-physical are
 * flagged and passed on, so that the drop happens on the Python side where it
 * is counted and reported.
 *
 * Returns null when no three.js scene can be found, which the caller treats as
 * "this page is not a 3D game" rather than as a failure.
 */
(function collectGameSceneProbe(options) {
  var config = options || {};
  var maxVerticesPerObject = config.maxVerticesPerObject || 200000;
  var excludeCameraDescendants = config.excludeCameraDescendants !== false;
  var THREE = window.__BENCHMARK_THREE__ || window.THREE;

  function findScene() {
    // Load-time instrumentation is the only reliable source: a top-level `const`
    // in a classic script is not a window property, and a scene held as a class
    // field is closed over, so neither is reachable by scanning globals.
    var registered = window.__BENCHMARK_SCENES__ || [];
    var best = null;
    for (var i = 0; i < registered.length; i += 1) {
      var candidate = registered[i];
      if (!candidate || !candidate.isScene || !candidate.children) {
        continue;
      }
      // A game may build offscreen or menu scenes alongside the playfield;
      // the richest graph is the one carrying the level.
      if (best === null || candidate.children.length > best.children.length) {
        best = candidate;
      }
    }
    if (best !== null) {
      return best;
    }
    if (window.__BENCHMARK_SCENE__ && window.__BENCHMARK_SCENE__.isScene) {
      return window.__BENCHMARK_SCENE__;
    }
    for (var key in window) {
      try {
        var value = window[key];
        if (value && value.isScene && value.children && value.children.length) {
          return value;
        }
      } catch (error) {
        /* Cross-origin or getter-backed globals are skipped deliberately. */
      }
    }
    return null;
  }

  var scene = findScene();
  if (!scene || !THREE) {
    return null;
  }
  scene.updateMatrixWorld(true);

  function sanitize(text) {
    return String(text || "").replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  }

  /* Category resolution walks up the ancestry so that an annotation placed on a
   * container still labels the meshes inside it. The source is reported because
   * a name-derived label carries no benchmark semantics and callers must be able
   * to see how much of a scene is actually labelled. */
  function resolveCategory(node, ancestors) {
    var chain = ancestors.concat([node]);
    for (var i = chain.length - 1; i >= 0; i -= 1) {
      var data = chain[i].userData;
      var explicit = data && (data.benchmarkCategory || data.category);
      if (explicit) {
        return { category: String(explicit), source: "declared" };
      }
    }
    if (node.name) {
      return { category: String(node.name).toLowerCase(), source: "node_name" };
    }
    for (var j = ancestors.length - 1; j >= 0; j -= 1) {
      if (ancestors[j].name) {
        return { category: String(ancestors[j].name).toLowerCase(), source: "ancestor_name" };
      }
    }
    return { category: String(node.type || "object").toLowerCase(), source: "node_type" };
  }

  function resolveDeclaredEntityKind(node, ancestors) {
    var chain = ancestors.concat([node]);
    for (var i = chain.length - 1; i >= 0; i -= 1) {
      var data = chain[i].userData;
      if (data && data.benchmarkEntityKind) {
        return String(data.benchmarkEntityKind).toLowerCase();
      }
    }
    return null;
  }

  /*
   * Runtime role signals are deliberately an allow-list of property *names*.
   * Values are never serialized: userData commonly contains circular game
   * objects and may also contain unrelated application state. These keys are
   * already used by the supported corpus for raycast/hit ownership.
   */
  var ROLE_SIGNAL_KEYS = {
    bot: "actor",
    combatant: "actor",
    player: "actor",
    npc: "actor",
    agent: "actor",
    projectile: "transient",
    effect: "transient",
    helper: "helper",
    viewmodel: "helper",
  };

  function ownRoleSignals(node) {
    var data = node && node.userData;
    var result = [];
    if (!data || typeof data !== "object") {
      return result;
    }
    Object.keys(ROLE_SIGNAL_KEYS).forEach(function inspect(key) {
      if (Object.prototype.hasOwnProperty.call(data, key) && data[key] != null) {
        result.push(key);
      }
    });
    return result;
  }

  var subtreeRole = new WeakMap();

  function analyzeSubtree(node) {
    var signals = ownRoleSignals(node);
    var meshCount = node && node.isMesh ? 1 : 0;
    var children = (node && node.children) || [];
    for (var i = 0; i < children.length; i += 1) {
      var child = analyzeSubtree(children[i]);
      meshCount += child.meshCount;
      for (var j = 0; j < child.signals.length; j += 1) {
        if (signals.indexOf(child.signals[j]) < 0) {
          signals.push(child.signals[j]);
        }
      }
    }
    var info = { signals: signals, meshCount: meshCount };
    subtreeRole.set(node, info);
    return info;
  }
  analyzeSubtree(scene);

  function familyBoundsPlausible(node) {
    try {
      var box = new THREE.Box3().setFromObject(node);
      if (box.isEmpty()) {
        return false;
      }
      var size = new THREE.Vector3();
      box.getSize(size);
      var scale = Number(config.unitScale || 1.0);
      var maximumFamilySpan = 8.0 / Math.max(scale, 1.0e-9);
      return Math.max(size.x, size.y, size.z) <= maximumFamilySpan;
    } catch (error) {
      return false;
    }
  }

  function roleFamily(node, ancestors) {
    var chain = ancestors.concat([node]).reverse();
    for (var i = 0; i < chain.length; i += 1) {
      var candidate = chain[i];
      if (!candidate || candidate.isScene) {
        continue;
      }
      var info = subtreeRole.get(candidate);
      if (!info || !info.signals.length || info.meshCount > 32) {
        continue;
      }
      if (!familyBoundsPlausible(candidate)) {
        continue;
      }
      return { node: candidate, signals: info.signals.slice() };
    }
    return null;
  }

  function resolveRuntimeRole(node, ancestors, path) {
    var declared = resolveDeclaredEntityKind(node, ancestors);
    var family = roleFamily(node, ancestors);
    var signals = family ? family.signals : ownRoleSignals(node);
    var signalKinds = signals.map(function mapSignal(key) {
      return ROLE_SIGNAL_KEYS[key];
    });
    var classification = declared || "static";
    var source = declared ? "declared_benchmark_entity_kind" : "default_static";
    if (signalKinds.indexOf("actor") >= 0) {
      classification = "dynamic_actor";
      source = declared === "static"
        ? "runtime_actor_signal_overrode_declared_static"
        : "runtime_actor_family_signal";
    } else if (signalKinds.indexOf("transient") >= 0) {
      classification = "transient_helper";
      source = "runtime_transient_family_signal";
    } else if (signalKinds.indexOf("helper") >= 0) {
      classification = "helper";
      source = "runtime_helper_family_signal";
    }
    var familyPath = null;
    if (family && family.node !== node) {
      var familyIndex = ancestors.indexOf(family.node);
      // ``ancestors`` includes the Scene at index 0 whereas ``path`` starts at
      // the Scene's selected child.  The graph path of ancestors[i] therefore
      // contains exactly i entries, not i + 1.
      familyPath = familyIndex >= 0 ? path.slice(0, familyIndex) : null;
    } else if (family) {
      familyPath = path.slice();
    }
    return {
      classification: classification,
      source: source,
      declared_entity_kind: declared,
      signal_keys: signals.slice().sort(),
      family_graph_path: familyPath,
    };
  }

  function materialVisibility(material) {
    var materials = Array.isArray(material) ? material : [material];
    var maximumOpacity = 0.0;
    var anyColorWrite = false;
    var anyMaterialVisible = false;
    for (var i = 0; i < materials.length; i += 1) {
      var item = materials[i];
      if (!item || item.visible === false) {
        continue;
      }
      var opacity = typeof item.opacity === "number" ? item.opacity : 1.0;
      maximumOpacity = Math.max(maximumOpacity, opacity);
      if (item.colorWrite !== false) {
        anyColorWrite = true;
      }
      if (opacity > 1.0e-4 && item.colorWrite !== false) {
        anyMaterialVisible = true;
      }
    }
    return {
      material_count: materials.length,
      maximum_opacity: maximumOpacity,
      any_color_write: anyColorWrite,
      any_material_visible: anyMaterialVisible,
    };
  }

  /* A mesh whose every material draws only back faces is invisible from outside
   * its own hull. That is a rendering idiom (inflated outline shells, skyboxes,
   * inward-facing room shells), not a solid, but the idiom covers both scenery
   * and real level boundaries, and material alone cannot tell them apart. The
   * probe therefore reports the hint and leaves the decision to the exporter,
   * which has the fitted boxes needed to tell an enclosed shell from an
   * enclosing one. */
  function backSideOnly(material) {
    var materials = Array.isArray(material) ? material : [material];
    if (!materials.length) {
      return false;
    }
    for (var i = 0; i < materials.length; i += 1) {
      if (!materials[i] || materials[i].side !== THREE.BackSide) {
        return false;
      }
    }
    return true;
  }

  var objects = [];
  var stats = {
    nodes_visited: 0,
    meshes_visited: 0,
    skipped_camera_descendant: 0,
    skipped_hidden: 0,
    skipped_empty_geometry: 0,
    flagged_back_side_only: 0,
    classified_dynamic_actor: 0,
    classified_transient_helper: 0,
    classified_helper: 0,
    emitted: 0,
  };

  var position = new THREE.Vector3();
  var quaternion = new THREE.Quaternion();
  var objectScale = new THREE.Vector3();
  var vertex = new THREE.Vector3();

  function bakeMesh(mesh) {
    var geometry = mesh.geometry;
    var attribute = geometry && geometry.attributes && geometry.attributes.position;
    if (!attribute || !attribute.count) {
      return null;
    }
    var vertices = [];
    var faces = [];
    var min = [Infinity, Infinity, Infinity];
    var max = [-Infinity, -Infinity, -Infinity];
    var truncated = attribute.count > maxVerticesPerObject;

    for (var i = 0; i < attribute.count; i += 1) {
      vertex.fromBufferAttribute(attribute, i).applyMatrix4(mesh.matrixWorld);
      for (var axis = 0; axis < 3; axis += 1) {
        var component = vertex.getComponent(axis);
        if (component < min[axis]) min[axis] = component;
        if (component > max[axis]) max[axis] = component;
      }
      if (!truncated) {
        vertices.push([vertex.x, vertex.y, vertex.z]);
      }
    }
    if (!isFinite(min[0]) || !isFinite(max[0])) {
      return null;
    }
    if (!truncated) {
      var indexAttribute = geometry.index;
      if (indexAttribute) {
        for (var t = 0; t + 2 < indexAttribute.count; t += 3) {
          faces.push([indexAttribute.getX(t), indexAttribute.getX(t + 1), indexAttribute.getX(t + 2)]);
        }
      } else {
        for (var v = 0; v + 2 < attribute.count; v += 3) {
          faces.push([v, v + 1, v + 2]);
        }
      }
    }
    return {
      world_bounds: { min: min, max: max },
      vertices: truncated ? [] : vertices,
      faces: truncated ? [] : faces,
      mesh_complete: !truncated && faces.length > 0,
    };
  }

  function walk(node, ancestors, path, visible) {
    stats.nodes_visited += 1;
    // First-person games parent the viewmodel weapon to the camera. It is
    // screen-space presentation, not level geometry, and neither it nor
    // anything below it may become a collision object.
    if (excludeCameraDescendants && node.isCamera) {
      stats.skipped_camera_descendant += 1;
      return;
    }
    var nodeVisible = visible && node.visible !== false;

    if (node.isMesh) {
      stats.meshes_visited += 1;
      if (!nodeVisible) {
        stats.skipped_hidden += 1;
      } else {
        var baked = bakeMesh(node);
        if (baked === null) {
          stats.skipped_empty_geometry += 1;
        } else {
          node.matrixWorld.decompose(position, quaternion, objectScale);
          var resolved = resolveCategory(node, ancestors);
          var runtimeRole = resolveRuntimeRole(node, ancestors, path);
          /*
           * The controlled benchmark cameras are rendered after this probe.
           * Leave only the coarse, benchmark-owned classification on the live
           * mesh so that those static-environment views can suppress actors and
           * helpers without re-running a second, potentially divergent role
           * detector.  No source userData values are copied.
           */
          try {
            node.userData = node.userData || {};
            node.userData.__benchmarkRuntimeRole = runtimeRole.classification;
          } catch (error) {
            /* A frozen userData object only loses visual suppression metadata;
             * the exported geometry classification remains authoritative. */
          }
          var visibility = materialVisibility(node.material);
          var nonPhysical = backSideOnly(node.material);
          if (nonPhysical) {
            stats.flagged_back_side_only += 1;
          }
          if (runtimeRole.classification === "dynamic_actor") {
            stats.classified_dynamic_actor += 1;
          } else if (runtimeRole.classification === "transient_helper") {
            stats.classified_transient_helper += 1;
          } else if (runtimeRole.classification === "helper") {
            stats.classified_helper += 1;
          }
          var trail = path.length ? path.join("_") : "0";
          var label = sanitize(node.name || node.type || "mesh") || "mesh";
          var sourceNames = [String(node.name || node.type)];
          for (var ancestorIndex = ancestors.length - 1; ancestorIndex >= 0; ancestorIndex -= 1) {
            var ancestorName = String(ancestors[ancestorIndex].name || "").trim();
            if (ancestorName && sourceNames.indexOf(ancestorName) < 0) {
              sourceNames.push(ancestorName);
            }
            if (sourceNames.length >= 8) {
              break;
            }
          }
          objects.push({
            id: label + "__" + trail,
            category: resolved.category,
            category_source: resolved.source,
            entity_kind: runtimeRole.classification,
            runtime_role: runtimeRole,
            material_visibility: visibility,
            rotation_quaternion: [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
            source_names: sourceNames,
            graph_path: path.slice(),
            graph_depth: path.length,
            non_physical_hint: nonPhysical ? "back_side_only" : null,
            world_bounds: baked.world_bounds,
            vertices: baked.vertices,
            faces: baked.faces,
            mesh_complete: baked.mesh_complete,
          });
          stats.emitted += 1;
        }
      }
    }

    var children = node.children || [];
    var nextAncestors = ancestors.concat([node]);
    for (var i = 0; i < children.length; i += 1) {
      walk(children[i], nextAncestors, path.concat([i]), nodeVisible);
    }
  }

  var topLevel = scene.children || [];
  for (var c = 0; c < topLevel.length; c += 1) {
    walk(topLevel[c], [scene], [c], true);
  }

  if (!objects.length) {
    return null;
  }

  return {
    schema_version: "game_scene_probe_v1",
    up_axis: "y",
    unit_scale: config.unitScale || 1.0,
    captured_at_tick: config.tick === undefined ? null : config.tick,
    deterministic_seed: config.seed === undefined ? null : config.seed,
    individualization: {
      strategy: "visible_mesh_runtime_role_v2",
      // Reported so that the count this strategy produces can be read against
      // the count a top-level-group strategy would have produced.
      top_level_child_count: topLevel.length,
      max_graph_depth: objects.reduce(function deepest(best, entry) {
        return entry.graph_depth > best ? entry.graph_depth : best;
      }, 0),
      declared_category_count: objects.filter(function declared(entry) {
        return entry.category_source === "declared";
      }).length,
      counts: stats,
    },
    objects: objects,
  };
})
