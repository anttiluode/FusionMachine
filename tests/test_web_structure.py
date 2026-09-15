from pathlib import Path


def test_root_page_exposes_the_fusion_machine_controls_and_panels():
    html = Path("index.html").read_text()
    assert 'href="web/style.css"' in html
    assert 'src="web/app.js"' in html
    for control in ("x0", "x1", "x2", "context"):
        assert f'id="{control}"' in html
    for panel in ("mode-a", "mode-b", "collapsed", "publication", "linear-attacker"):
        assert f'id="{panel}"' in html


def test_page_copy_keeps_the_scientific_claim_boundary_visible():
    html = Path("index.html").read_text()
    assert "same answer, different algorithm" in html.lower()
    assert "not a biological-neuron claim" in html.lower()
