(() => {
  const $ = (id) => document.getElementById(id);
  const controls = ["x0", "x1", "x2", "context"];
  let timer = null;
  let playIndex = 0;

  function n(id) { return Number($(id).value); }
  function algorithmA(x0) { return x0; }
  function algorithmB(x1, x2) { return x1 * x2; }
  function nonlinear(a, b, c) { return 0.5 * (a + b) + 0.5 * c * (b - a); }
  function linear(a, b) { return 0.5 * (a + b); }
  function fmt(v) { return Number(v).toFixed(1); }

  function rows() {
    const out = [];
    for (const x0 of [-1, 1]) for (const x1 of [-1, 1]) for (const x2 of [-1, 1]) for (const c of [-1, 1]) {
      const a = algorithmA(x0);
      const b = algorithmB(x1, x2);
      out.push({ x0, x1, x2, c, a, b, target: c === -1 ? a : b, nonlinear: nonlinear(a, b, c), linear: linear(a, b) });
    }
    return out;
  }

  function current() {
    const x0 = n("x0"), x1 = n("x1"), x2 = n("x2"), c = n("context");
    const a = algorithmA(x0), b = algorithmB(x1, x2);
    return { x0, x1, x2, c, a, b, target: c === -1 ? a : b, nonlinear: nonlinear(a, b, c), linear: linear(a, b) };
  }

  function renderTable(state) {
    $("truth-body").innerHTML = rows().map(r => {
      const same = r.x0 === state.x0 && r.x1 === state.x1 && r.x2 === state.x2 && r.c === state.c;
      const cls = `${same ? "current " : ""}${r.a !== r.b ? "disagree" : ""}`;
      return `<tr class="${cls.trim()}"><td>${r.x0}</td><td>${r.x1}</td><td>${r.x2}</td><td>${r.c}</td><td>${r.a}</td><td>${r.b}</td><td>${r.target}</td><td>${r.nonlinear}</td><td>${fmt(r.linear)}</td></tr>`;
    }).join("");
  }

  function render() {
    const s = current();
    $("x0-out").textContent = s.x0;
    $("x1-out").textContent = s.x1;
    $("x2-out").textContent = s.x2;
    $("context-out").textContent = s.c === -1 ? "A" : "B";
    $("a-value").textContent = s.a > 0 ? "+1" : "−1";
    $("b-value").textContent = s.b > 0 ? "+1" : "−1";
    $("sum-term").textContent = fmt(0.5 * (s.a + s.b));
    $("gate-term").textContent = fmt(0.5 * s.c * (s.b - s.a));
    $("published-value").textContent = s.nonlinear > 0 ? "+1" : "−1";
    $("publication-label").textContent = `publishing ${s.c === -1 ? "A · direct" : "B · relational"}`;
    $("collapsed-value").textContent = fmt(0.5 * (s.a + s.b));
    $("linear-value").textContent = fmt(s.linear);

    $("mode-a").classList.toggle("active", s.c === -1);
    $("mode-b").classList.toggle("active", s.c === 1);

    const correlated = s.a === s.b;
    $("world-status").innerHTML = correlated
      ? `<strong>Correlated world:</strong> A and B agree (${s.a}). Behavior cannot reveal which route produced it.`
      : `<strong>Intervention:</strong> A=${s.a}, B=${s.b}. The hidden algorithms now disagree, so context has real work to do.`;
    $("collapse-explain").textContent = correlated
      ? "Here collapse is harmless because both algorithms happen to agree."
      : "A and B disagree, so collapse produces 0 and erases which algorithm carried +1 versus −1.";

    renderTable(s);
  }

  function setState(r) {
    $("x0").value = r.x0;
    $("x1").value = r.x1;
    $("x2").value = r.x2;
    $("context").value = r.c;
    render();
  }

  controls.forEach(id => $(id).addEventListener("input", render));
  $("correlate").addEventListener("click", () => {
    $("x0").value = n("x1") * n("x2");
    render();
  });
  $("autoplay").addEventListener("click", () => {
    if (timer) {
      clearInterval(timer);
      timer = null;
      $("autoplay").textContent = "Auto-play truth table";
      return;
    }
    const all = rows();
    $("autoplay").textContent = "Pause auto-play";
    timer = setInterval(() => {
      setState(all[playIndex % all.length]);
      playIndex += 1;
    }, 850);
  });

  render();
})();
