(() => {
  const $ = (id) => document.getElementById(id);
  const gap = 80;

  function analyticRmse(alpha, k) {
    if (k >= gap) return 0;
    const variance =
      ((1 - alpha) ** 2) *
      (alpha ** (2 * k)) *
      (1 - alpha ** (2 * (gap - k))) /
      (1 - alpha ** 2);
    return Math.sqrt(Math.max(0, variance));
  }

  function horizon(alpha, fraction = 0.1) {
    const base = analyticRmse(alpha, 0);
    for (let k = 0; k <= gap; k += 1) {
      if (analyticRmse(alpha, k) <= fraction * base) return k;
    }
    return gap;
  }

  function renderTape(k) {
    const tape = $("v1-tape");
    tape.innerHTML = "";
    for (let i = 0; i < gap; i += 1) {
      const cell = document.createElement("span");
      const remembered = i >= gap - k;
      cell.className = remembered ? "remembered" : "forgotten";
      cell.title = remembered ? `step ${i + 1}: retained for replay` : `step ${i + 1}: omitted`;
      tape.appendChild(cell);
    }
  }

  function render() {
    const alpha = Number($("v1-alpha").value);
    const k = Number($("v1-k").value);
    const base = analyticRmse(alpha, 0);
    const rmse = analyticRmse(alpha, k);
    const ratio = base > 0 ? rmse / base : 0;
    const needed = horizon(alpha, 0.1);

    $("v1-alpha-out").textContent = alpha.toFixed(2);
    $("v1-k-out").textContent = String(k);
    $("v1-rmse").textContent = rmse.toFixed(6);
    $("v1-ratio").textContent = `${(100 * ratio).toFixed(1)}%`;
    $("v1-horizon").textContent = `${needed} / ${gap}`;
    $("v1-switch-cost").textContent = k === gap ? "exact replay" : k === 0 ? "no history" : `${k} replay updates`;

    const resident = $("v1-resident-status");
    resident.innerHTML = `<strong>Resident route:</strong> ${gap} updates paid during the gap → <em>0 replay updates at switch</em>.`;

    const lazy = $("v1-lazy-status");
    lazy.innerHTML = k === gap
      ? `<strong>Lazy route:</strong> stores all ${gap} drives and replays all ${gap} → <em>exact state</em>.`
      : `<strong>Lazy route:</strong> stores ${k} recent drives → omitted history leaves predicted switch RMSE <em>${rmse.toFixed(6)}</em>.`;

    renderTape(k);
  }

  $("v1-alpha").addEventListener("input", render);
  $("v1-k").addEventListener("input", render);
  render();
})();
