from pathlib import Path


def test_root_page_exposes_the_fusion_machine_controls_and_panels():
    html = Path("index.html").read_text()
    assert 'href="web/style.css"' in html
    assert 'href="web/v3.css"' in html
    assert 'src="web/app.js"' in html
    assert 'src="web/v1.js"' in html
    assert 'src="web/v2.js"' in html
    for control in ("x0", "x1", "x2", "context"):
        assert f'id="{control}"' in html
    for panel in ("mode-a", "mode-b", "collapsed", "publication", "linear-attacker"):
        assert f'id="{panel}"' in html


def test_v1_page_exposes_resident_replay_controls_and_readiness_accounting():
    html = Path("index.html").read_text()
    assert 'id="v1-lab"' in html
    assert 'id="v1-alpha"' in html
    assert 'id="v1-k"' in html
    assert 'id="v1-tape"' in html
    assert 'id="v1-rmse"' in html
    assert 'id="v1-horizon"' in html
    assert "persistence is also a replay horizon" in html.lower()
    assert "full replay is explicitly exact" in html.lower()


def test_v2_page_exposes_dendrite_ais_axon_routing_lab():
    html = Path("index.html").read_text()
    for element in (
        "v2-lab",
        "v2-source",
        "v2-ais-gate",
        "v2-pulse",
        "v2-route-mode",
        "v2-targets",
        "v2-reroute",
        "v2-step",
    ):
        assert f'id="{element}"' in html
    assert "one-bit event" in html.lower()
    assert "address lives in the route" in html.lower()


def test_v3_page_exposes_fair_fight_and_confound():
    html = Path("index.html").read_text().lower()
    for element in ("v3-lab", "v3-budget", "v3-controls"):
        assert f'id="{element}"' in html
    assert "four states vs four states" in html
    assert "state is not relevance" in html
    assert "confounded" in html
    assert "30 vs 46" in html
    assert "1,088" in html


def test_page_copy_keeps_the_scientific_claim_boundary_visible():
    html = Path("index.html").read_text().lower()
    assert "same answer" in html
    assert "not biological-neuron claims" in html
    assert "not evidence that basket/chandelier cells implement" in html
