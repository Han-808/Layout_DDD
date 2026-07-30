/*
 * Deterministic replay harness, injected before any page script runs.
 *
 * The benchmark hash-freezes its inputs and compares submissions against each
 * other, so a scene export that differs between two runs of the same build is
 * worthless. Three sources of run-to-run drift are removed here: the random
 * number generator, wall-clock time, and frame pacing. After injection the page
 * only advances when the driver calls __benchmarkStep(), and each step advances
 * the clock by exactly the configured delta.
 */
(function installDeterministicHarness(options) {
  var config = options || {};
  var seed = (config.seed === undefined ? 20260727 : config.seed) >>> 0;
  var stepMs = config.stepMs || 1000 / 60;
  var epochMs = config.epochMs || 0;

  var randomState = seed || 1;
  Math.random = function deterministicRandom() {
    // xorshift32: small, seedable, and identical across engines.
    randomState ^= randomState << 13;
    randomState ^= randomState >>> 17;
    randomState ^= randomState << 5;
    randomState >>>= 0;
    return randomState / 4294967296;
  };

  var virtualNow = 0;
  var nativeDateNow = Date.now.bind(Date);
  performance.now = function deterministicPerformanceNow() {
    return virtualNow;
  };
  Date.now = function deterministicDateNow() {
    return epochMs + virtualNow;
  };
  window.__benchmarkNativeDateNow = nativeDateNow;

  var pending = [];
  var nextHandle = 1;
  window.requestAnimationFrame = function deterministicRaf(callback) {
    var handle = nextHandle;
    nextHandle += 1;
    pending.push({ handle: handle, callback: callback });
    return handle;
  };
  window.cancelAnimationFrame = function deterministicCancelRaf(handle) {
    pending = pending.filter(function keep(entry) {
      return entry.handle !== handle;
    });
  };

  window.__benchmarkStep = function step(frames) {
    var count = frames || 1;
    for (var i = 0; i < count; i += 1) {
      virtualNow += stepMs;
      var due = pending;
      pending = [];
      for (var j = 0; j < due.length; j += 1) {
        try {
          due[j].callback(virtualNow);
        } catch (error) {
          window.__benchmarkStepError = String((error && error.stack) || error);
        }
      }
    }
    return { tick: Math.round(virtualNow / stepMs), virtual_now_ms: virtualNow };
  };

  window.__benchmarkDeterminism = {
    seed: seed,
    step_ms: stepMs,
    epoch_ms: epochMs,
    installed: true,
  };
})
