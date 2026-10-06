import pytest

tf = pytest.importorskip("tensorflow")


def test_architecture_matches_paper():
    from gujbraille.model import build_cnn
    m = build_cnn()
    assert len(m.layers) == 9
    assert m.count_params() == 117440
    assert m.output_shape == (None, 64)
    assert [l.output.shape[1:] for l in m.layers][:6] == [
        (26, 26, 32), (13, 13, 32), (11, 11, 64), (5, 5, 64), (3, 3, 128), (1, 1, 128)]
