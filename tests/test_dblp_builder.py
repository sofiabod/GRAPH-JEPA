import io
import gzip

import torch
from torch_geometric.data import Data

from src.data.dblp_builder import (
    build_dblp_graphs_from_records,
    _iter_papers,
)


def make_fake_records(n_authors=20, n_years=15, papers_per_year=40,
                      avg_authors_per_paper=3, year_start=2000, seed=0):
    """generate fake (year, [author_name]) tuples with realistic per-author counts."""
    import random
    rng = random.Random(seed)
    author_names = [f"author_{i:03d}" for i in range(n_authors)]

    records = []
    author_paper_counts = {}
    for a in author_names:
        author_paper_counts[a] = 0

    for y_idx in range(n_years):
        year = year_start + y_idx
        for _ in range(papers_per_year):
            n_a = max(1, int(rng.gauss(avg_authors_per_paper, 1.0)))
            n_a = min(n_a, n_authors)
            authors = rng.sample(author_names, n_a)
            records.append((year, authors))
            for a in authors:
                author_paper_counts[a] += 1
    return records, author_paper_counts


def test_output_is_list_of_data():
    records, counts = make_fake_records()
    graphs, meta = build_dblp_graphs_from_records(
        records, counts, min_papers=1, min_active_nodes=2,
    )
    assert isinstance(graphs, list)
    assert len(graphs) > 0
    assert isinstance(graphs[0], Data)


def test_node_feature_dim_is_6():
    records, counts = make_fake_records()
    graphs, meta = build_dblp_graphs_from_records(
        records, counts, min_papers=1, min_active_nodes=2,
    )
    for g in graphs:
        assert g.x.shape[1] == 6, f"expected in_dim=6, got {g.x.shape[1]}"


def test_edge_index_valid():
    records, counts = make_fake_records()
    graphs, meta = build_dblp_graphs_from_records(
        records, counts, min_papers=1, min_active_nodes=2,
    )
    for g in graphs:
        assert g.edge_index.shape[0] == 2
        if g.edge_index.shape[1] > 0:
            assert g.edge_index.max() < g.x.shape[0]


def test_meta_has_split_ranges():
    records, counts = make_fake_records(n_years=20)
    graphs, meta = build_dblp_graphs_from_records(
        records, counts, min_papers=1, min_active_nodes=2,
    )
    assert "train_range" in meta and "val_range" in meta and "test_range" in meta
    assert meta["dataset"] == "dblp"
    assert meta["train_range"][0] == 0
    assert meta["test_range"][1] == meta["n_snapshots"] - 1


def test_volume_feature_normalized():
    records, counts = make_fake_records()
    graphs, meta = build_dblp_graphs_from_records(
        records, counts, min_papers=1, min_active_nodes=2,
    )
    for g in graphs:
        # col 0 = normalized publication count, must be in [0, 1]
        assert g.x[:, 0].min() >= 0.0
        assert g.x[:, 0].max() <= 1.0 + 1e-6


def test_min_papers_filter_drops_low_count_authors():
    # 10 prolific authors on every paper, 10 truly one-shot authors
    records = []
    counts = {}
    prolific = [f"a_{i}" for i in range(10)]
    one_shot = [f"b_{i}" for i in range(10)]
    for a in prolific + one_shot:
        counts[a] = 0
    # one paper per one-shot author, all years collapsed onto a single paper
    for i, b in enumerate(one_shot):
        authors = prolific + [b]
        records.append((2000 + i, authors))
        for a in authors:
            counts[a] += 1
    graphs, meta = build_dblp_graphs_from_records(
        records, counts, min_papers=5, min_active_nodes=2,
    )
    # only prolific authors survive (one_shot have count 1 < 5)
    assert meta["n_nodes"] == 10


def test_max_authors_caps_node_set():
    records, counts = make_fake_records(n_authors=50)
    graphs, meta = build_dblp_graphs_from_records(
        records, counts, min_papers=1, min_active_nodes=2, max_authors=20,
    )
    assert meta["n_nodes"] == 20
    for g in graphs:
        assert g.x.shape[0] == 20


def test_min_active_nodes_filter():
    # one paper per year with only 2 authors -> 2 active nodes
    records = []
    counts = {"x": 0, "y": 0}
    for year in range(2000, 2010):
        records.append((year, ["x", "y"]))
        counts["x"] += 1
        counts["y"] += 1
    graphs, meta = build_dblp_graphs_from_records(
        records, counts, min_papers=1, min_active_nodes=10,
    )
    assert len(graphs) == 0
    assert meta["n_snapshots"] == 0


def test_edge_index_symmetric():
    # each coauthorship should appear in both directions
    records = [(2000, ["a", "b", "c"])]
    counts = {"a": 1, "b": 1, "c": 1}
    graphs, meta = build_dblp_graphs_from_records(
        records, counts, min_papers=1, min_active_nodes=2,
    )
    assert len(graphs) == 1
    g = graphs[0]
    # 3 unordered pairs = 6 directed edges
    assert g.edge_index.shape[1] == 6


def test_iter_papers_streaming_xml_gz():
    # build a tiny in-memory xml.gz, stream-parse it back
    xml = (
        '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
        '<dblp>\n'
        '<article key="x/y/1"><author>Alice</author><author>Bob</author>'
        '<title>T1</title><year>2010</year></article>\n'
        '<inproceedings key="x/y/2"><author>Bob</author><author>Carol</author>'
        '<title>T2</title><year>2011</year></inproceedings>\n'
        '<article key="x/y/3"><author>Alice</author>'
        '<title>T3</title><year>1990</year></article>\n'
        '</dblp>\n'
    )
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(xml.encode("utf-8"))
    buf.seek(0)

    # write to a tmp file, _iter_papers takes a path
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".xml.gz", delete=False) as f:
        f.write(buf.getvalue())
        tmp_path = f.name

    out = list(_iter_papers(tmp_path, year_min=2000, year_max=2020))
    # 1990 paper out of range; 2 papers in range
    assert len(out) == 2
    years = [y for y, _ in out]
    assert sorted(years) == [2010, 2011]
