from pathlib import Path


def test_root_page_exposes_the_fusion_machine_controls_and_panels():
    html = Path("index.html").read_text()
    assert 'href="web/style.css"' in html
    assert 'src="web/app.js"' in html
    assert 'src="web/v1.js"' in html
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


def test_page_copy_keeps_the_scientific_claim_boundary_visible():
    html = Path("index.html").read_text()
    assert "same answer" in html.lower()
    assert "not a biological-neuron claim" in html.lower()
