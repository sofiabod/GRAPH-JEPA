import torch

from src.models.predictor import TemporalGraphPredictor


# small predictor config for attention tests
SMALL_KWARGS = dict(embed_dim=32, n_heads=4, n_layers=2, n_nodes=10, max_time_steps=20)


def make_small_predictor(**kwargs):
    defaults = dict(SMALL_KWARGS)
    defaults.update(kwargs)
    return TemporalGraphPredictor(**defaults)


def test_output_shape():
    predictor = TemporalGraphPredictor()
    B, T, D = 2, 6, 256
    tokens = torch.randn(B, T, D)
    time_indices = torch.zeros(B, T, dtype=torch.long)
    node_ids = torch.zeros(B, T, dtype=torch.long)
    out = predictor(tokens, time_indices, node_ids)
    assert out.shape == (B, T, D), f"expected [{B}, {T}, {D}], got {out.shape}"


def test_bidirectional_attention():
    # change token at position 0, verify output at position 1 changes
    predictor = make_small_predictor()
    B, T, D = 1, 4, 32
    tokens = torch.randn(B, T, D)
    time_indices = torch.zeros(B, T, dtype=torch.long)
    node_ids = torch.zeros(B, T, dtype=torch.long)

    out1 = predictor(tokens, time_indices, node_ids)

    tokens2 = tokens.clone()
    tokens2[0, 0] = tokens2[0, 0] + 10.0  # big change at position 0
    out2 = predictor(tokens2, time_indices, node_ids)

    # position 1 should change because attention is bidirectional
    assert not torch.allclose(out1[0, 1], out2[0, 1], atol=1e-4), \
        "output at position 1 should change when position 0 token changes (bidirectional attention)"


def test_full_context_changes_mask_output():
    # change a non-mask (visible) token, verify mask position output changes
    predictor = make_small_predictor()
    B, T, D = 1, 5, 32
    tokens = torch.randn(B, T, D)
    time_indices = torch.zeros(B, T, dtype=torch.long)
    node_ids = torch.zeros(B, T, dtype=torch.long)

    out1 = predictor(tokens, time_indices, node_ids)

    tokens2 = tokens.clone()
    tokens2[0, 0] = tokens2[0, 0] + 10.0  # change visible token at position 0
    out2 = predictor(tokens2, time_indices, node_ids)

    # mask token at last position should change
    mask_pos = T - 1
    assert not torch.allclose(out1[0, mask_pos], out2[0, mask_pos], atol=1e-4), \
        "mask token output should change when context token changes"


def test_node_id_embedding_matters():
    predictor = make_small_predictor()
    B, T, D = 1, 3, 32
    tokens = torch.randn(B, T, D)
    time_indices = torch.zeros(B, T, dtype=torch.long)

    node_ids1 = torch.zeros(B, T, dtype=torch.long)
    node_ids2 = node_ids1.clone()
    node_ids2[0, -1] = 5  # different node id at mask position

    out1 = predictor(tokens, time_indices, node_ids1)
    out2 = predictor(tokens, time_indices, node_ids2)

    assert not torch.allclose(out1[0, -1], out2[0, -1], atol=1e-4), \
        "different node ids at mask position should produce different outputs"


def test_temporal_pos_embedding_matters():
    predictor = make_small_predictor()
    B, T, D = 1, 3, 32
    tokens = torch.randn(B, T, D)
    node_ids = torch.zeros(B, T, dtype=torch.long)

    time_indices1 = torch.zeros(B, T, dtype=torch.long)
    time_indices2 = time_indices1.clone()
    time_indices2[0, -1] = 10  # different time index at mask position

    out1 = predictor(tokens, time_indices1, node_ids)
    out2 = predictor(tokens, time_indices2, node_ids)

    assert not torch.allclose(out1[0, -1], out2[0, -1], atol=1e-4), \
        "different time indices at mask position should produce different outputs"



def test_param_count():
    predictor = TemporalGraphPredictor()
    total = sum(p.numel() for p in predictor.parameters())
    assert 100_000 <= total <= 5_000_000, \
        f"param count {total} outside expected range [100_000, 5_000_000]"
