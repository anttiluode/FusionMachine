(() => {
  const css = document.createElement("link");
  css.rel = "stylesheet";
  css.href = "web/v2.css";
  document.head.appendChild(css);

  const $ = (id) => document.getElementById(id);
  const targetStates = [0, 0, 0, 0];
  let sourceState = 0;
  let stepIndex = 0;
  let rerouted = false;
  let timer = null;

  const maps = {
    learned: [0, 1, 2, 3],
    chemistry: [0, 0, 2, 2],
    activity: [0, 1, 0, 1],
    shuffled: [1, 2, 3, 0],
  };

  function selectedSource() { return Number($("v2-source").value); }
  function selectedMode() { return $("v2-route-mode").value; }

  function destination(source) {
    const base = maps[selectedMode()][source];
    return rerouted && source === selectedSource() ? (base + 1) % targetStates.length : base;
  }

  function renderTargets(activeTarget = -1) {
    document.querySelectorAll("#v2-targets [data-target]").forEach((node) => {
      const i = Number(node.dataset.target);
      node.classList.toggle("active", i === activeTarget);
      node.querySelector("strong").textContent = targetStates[i].toFixed(2);
    });
  }

  function renderStatic() {
    const source = selectedSource();
    const target = destination(source);
    $("v2-axon-id").textContent = `axon A${source}`;
    $("v2-route-label").textContent = `A${source} → D${target}${rerouted ? " · rerouted" : ""}`;
    $("v2-reroute").textContent = rerouted ? "Restore original branch" : "Reroute one axon";
  }

  function pulseAnimation() {
    const pulse = $("v2-moving-pulse");
    pulse.classList.remove("fly");
    void pulse.offsetWidth;
    pulse.classList.add("fly");
  }

  function log(text) {
    const item = document.createElement("li");
    item.textContent = text;
    $("v2-log").prepend(item);
    while ($("v2-log").children.length > 6) $("v2-log").lastElementChild.remove();
  }

  function step() {
    stepIndex += 1;
    const source = selectedSource();
    const drive = 0.72 + 0.06 * source;
    sourceState = 0.72 * sourceState + 0.28 * drive;
    const suppressed = $("v2-ais-gate").checked;
    const event = !suppressed && sourceState >= 0.42 ? 1 : 0;

    $("v2-source-state").textContent = sourceState.toFixed(2);
    $("v2-source-bar").style.width = `${Math.min(100, sourceState * 100)}%`;
    $("v2-pulse").textContent = String(event);
    $("v2-payload").textContent = `${event} · one-bit event`;
    $("v2-ais-orb").classList.toggle("firing", event === 1);
    $("v2-ais-orb").classList.toggle("suppressed", suppressed);

    if (suppressed) {
      $("v2-ais-status").textContent = "publication suppressed · resident state still updating";
      renderTargets(-1);
      log(`t${stepIndex}: A${source} resident=${sourceState.toFixed(2)} · AIS blocked · no traffic`);
      return;
    }

    if (event === 0) {
      $("v2-ais-status").textContent = "resident state below commit threshold";
      renderTargets(-1);
      log(`t${stepIndex}: A${source} resident=${sourceState.toFixed(2)} · no event`);
      return;
    }

    const target = destination(source);
    targetStates[target] = 0.65 * targetStates[target] + 0.35;
    $("v2-ais-status").textContent = `commit: one bit leaves as axon A${source}`;
    renderTargets(target);
    pulseAnimation();
    log(`t${stepIndex}: payload=1 · route A${source}→D${target} · D${target}=${targetStates[target].toFixed(2)}`);
  }

  $("v2-source").addEventListener("change", () => {
    sourceState = 0;
    renderStatic();
    renderTargets(-1);
  });
  $("v2-route-mode").addEventListener("change", renderStatic);
  $("v2-ais-gate").addEventListener("change", () => {
    renderStatic();
    $("v2-ais-status").textContent = $("v2-ais-gate").checked
      ? "publication suppression armed"
      : "publication gate open";
  });
  $("v2-reroute").addEventListener("click", () => {
    rerouted = !rerouted;
    renderStatic();
    log(rerouted ? "material intervention: branch destination changed" : "material intervention reversed");
  });
  $("v2-step").addEventListener("click", step);
  $("v2-autoplay").addEventListener("click", () => {
    if (timer) {
      clearInterval(timer);
      timer = null;
      $("v2-autoplay").textContent = "Auto-play";
      return;
    }
    timer = setInterval(step, 750);
    $("v2-autoplay").textContent = "Pause";
  });

  renderStatic();
  renderTargets(-1);
})();
