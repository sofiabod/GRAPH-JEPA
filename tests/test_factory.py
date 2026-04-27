import json
import torch
from pathlib import Path
from types import SimpleNamespace

from src.data.factory import create_dataset, compute_split_ranges, save_meta


def test_compute_split_ranges_proportions():
    train, val, test = compute_split_ranges(100)
    assert train == (0, 69)
    assert val == (70, 84)
    assert test == (85, 99)
    # all ranges together cover 0..99 with no overlap
    assert train[1] + 1 == val[0]
    assert val[1] + 1 == test[0]
    assert test[1] == 99


def test_compute_split_ranges_coverage():
    for n in [10, 50, 180, 200]:
        train, val, test = compute_split_ranges(n)
        assert train[0] == 0
        assert test[1] == n - 1
        assert train[1] < val[0]
        assert val[1] < test[0]


def test_save_meta_writes_json(tmp_path):
    meta = {"train_range": [0, 9], "val_range": [10, 14], "test_range": [15, 19],
            "n_snapshots": 20, "n_nodes": 5}
    out = str(tmp_path / "sub" / "meta.json")
    save_meta(meta, out)
    loaded = json.loads(Path(out).read_text())
    assert loaded["train_range"] == [0, 9]


def test_unknown_dataset_raises():
    cfg = SimpleNamespace(dataset="unknown_xyz", data=SimpleNamespace())
    try:
        create_dataset(cfg)
        assert False, "should have raised"
    except ValueError as e:
        assert "unknown_xyz" in str(e)


def test_enron_route_loads_graphs(tmp_path):
    # create a fake .pt file and verify create_dataset returns it
    # use a plain list of tensors (no custom class needed)
    graphs = [{"x": torch.zeros(3, 5)}]
    pt_path = str(tmp_path / "g.pt")
    torch.save(graphs, pt_path)

    cfg = SimpleNamespace(dataset="enron", data=SimpleNamespace(
        graphs_path=pt_path,
        train_weeks=[0, 6],
    ))
    result = create_dataset(cfg)
    assert len(result) == 1
