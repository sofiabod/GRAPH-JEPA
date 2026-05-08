"""builder for dblp computer science coauthorship temporal graph.

source: https://dblp.org/xml/dblp.xml.gz (full release dump from dblp.org)

shape: same as baci_gravity. one snapshot per year, fixed author node set
filtered to authors with at least min_papers across the window.
node features = [1d normalized total publication count, 5d structural] = 6d.

each paper contributes a clique of coauthor edges in that paper's year. edge
weight is the number of joint papers in that year.
"""
from collections import defaultdict
from pathlib import Path
import gzip
import io
import re
import urllib.request
import xml.etree.ElementTree as ET
from html.entities import name2codepoint

import torch
from torch_geometric.data import Data

from src.data.graph_utils import compute_structural_features
from src.data.factory import compute_split_ranges


# precompiled regex to substitute named html entities for numeric refs.
# the dblp xml dtd does not declare html entities like &uuml; etc. in its dtd,
# but the data uses them heavily. converting to &#NNN; lets the standard
# xml parser handle them without dtd resolution.
_ENT_RE = re.compile(rb"&([A-Za-z][A-Za-z0-9]+);")


def _entity_sub(m: "re.Match") -> bytes:
    name = m.group(1).decode("ascii", errors="ignore")
    if name in ("amp", "lt", "gt", "quot", "apos"):
        return m.group(0)  # leave xml builtins alone
    cp = name2codepoint.get(name)
    if cp is None:
        return m.group(0)  # unknown: pass through
    return f"&#{cp};".encode("ascii")


class _EntityFilteredStream:
    """wrap a binary stream and substitute html entity refs on the fly.

    keeps a small tail buffer so a multi-byte entity split across read
    boundaries still gets matched on the next read. exposes only a `read(size)`
    method which is what xml.etree.ElementTree.iterparse expects.
    """

    def __init__(self, raw):
        self._raw = raw
        self._tail = b""
        self._pending = b""

    def read(self, size=-1):
        if size is None or size < 0:
            data = self._tail + self._raw.read()
            self._tail = b""
            return self._pending + _ENT_RE.sub(_entity_sub, data)
        # accumulate substituted output until we have at least `size` bytes
        while len(self._pending) < size:
            raw = self._raw.read(max(size, 65536))
            if not raw:
                # final flush
                final = _ENT_RE.sub(_entity_sub, self._tail)
                self._tail = b""
                self._pending += final
                break
            data = self._tail + raw
            # find a safe cut point that never splits an entity across reads.
            # entities like `&aacute;` are up to ~10 bytes wide; if we cut
            # blindly the `&` may end up in head and `;` in tail, leaving the
            # partial `&aacute` past end of head, which sub() can't match.
            # fix: search the boundary zone (last 80 bytes) for any `&` and
            # cut just before the earliest one found there. that guarantees
            # the entity ends up entirely in tail, and the next iteration
            # will see it in head with its closing `;`.
            base = max(0, len(data) - 80)
            amp_in_zone = data.find(b"&", base)
            if amp_in_zone >= 0:
                cut = amp_in_zone
            else:
                cut = max(0, len(data) - 64)
            head = data[:cut]
            self._tail = data[cut:]
            self._pending += _ENT_RE.sub(_entity_sub, head)
        out = self._pending[:size]
        self._pending = self._pending[size:]
        return out


DBLP_URL = "https://dblp.org/xml/dblp.xml.gz"


def download_dblp(data_dir: str) -> str:
    """download dblp xml dump if absent, return path to gzipped xml."""
    out = Path(data_dir) / "dblp_raw" / "dblp.xml.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        print(f"downloading dblp xml from {DBLP_URL}...")
        urllib.request.urlretrieve(DBLP_URL, out)
    return str(out)


# tags that count as publications with author + year metadata
PAPER_TAGS = ("article", "inproceedings", "incollection", "proceedings", "book", "phdthesis", "mastersthesis")


def _iter_papers(xml_gz_path: str, year_min: int, year_max: int):
    """stream parse the dblp xml.gz, yielding (year, [author_names]) per paper.

    uses iterparse with element clearing to keep memory bounded. wraps the
    gzip stream in a filter that converts html named entities (&uuml; etc.)
    to numeric refs so the stdlib xml parser can handle them.
    """
    with gzip.open(xml_gz_path, "rb") as gz:
        f = _EntityFilteredStream(gz)
        context = ET.iterparse(f, events=("start", "end"))
        _, root = next(context)
        for event, elem in context:
            if event != "end":
                continue
            tag = elem.tag
            if tag in PAPER_TAGS:
                year_text = None
                authors = []
                for child in elem:
                    if child.tag == "year" and child.text:
                        year_text = child.text.strip()
                    elif child.tag == "author" and child.text:
                        authors.append(child.text.strip())
                if year_text and authors:
                    try:
                        year = int(year_text)
                    except ValueError:
                        year = None
                    if year is not None and year_min <= year <= year_max:
                        yield year, authors
                # free memory
                elem.clear()
                root.clear()


def _collect_paper_records(xml_gz_path: str, year_min: int, year_max: int):
    """first pass: collect (year, [authors]) for every paper in range.

    returns:
        records: list of (year, frozenset(authors)) for fast second pass
        author_paper_counts: dict author -> total paper count across window
    """
    records = []
    author_paper_counts = defaultdict(int)
    for year, authors in _iter_papers(xml_gz_path, year_min, year_max):
        # dedupe author list per paper to avoid counting duplicates
        unique = list(dict.fromkeys(authors))
        records.append((year, unique))
        for a in unique:
            author_paper_counts[a] += 1
    return records, author_paper_counts


def build_dblp_graphs_from_records(
    records,
    author_paper_counts,
    min_papers: int = 5,
    min_active_nodes: int = 30,
    max_authors: int = None,
):
    """build annual coauthorship graph snapshots from paper records.

    args:
        records: list of (year, [author_name]) tuples
        author_paper_counts: dict author -> total paper count across window
        min_papers: keep authors with >= min_papers across the window
        min_active_nodes: drop years with fewer active kept authors
        max_authors: optional cap on the number of authors (top by paper count)

    each snapshot:
        x: [N, 6] (1d normalized incident publication count, 5d structural)
        edge_index: [2, E] symmetric directed (each coauthorship adds both directions)
        edge_attr: [E, 1] (joint paper count)
        node_ids: arange(N)

    returns (graphs, meta)
    """
    # author filter: keep those with >= min_papers across the full window
    kept = [a for a, c in author_paper_counts.items() if c >= min_papers]
    if max_authors is not None and len(kept) > max_authors:
        # rank by paper count, keep top max_authors
        kept = sorted(kept, key=lambda a: author_paper_counts[a], reverse=True)[:max_authors]
    kept_set = set(kept)
    authors_sorted = sorted(kept_set)
    author2id = {a: i for i, a in enumerate(authors_sorted)}
    n_nodes = len(author2id)

    # accumulate joint paper counts and per-year per-author paper counts
    by_year_pair = defaultdict(lambda: defaultdict(float))  # year -> (s, d) -> count
    by_year_author_papers = defaultdict(lambda: defaultdict(int))  # year -> author_id -> paper count
    for year, authors in records:
        kept_auth = [a for a in authors if a in kept_set]
        if not kept_auth:
            continue
        ids = [author2id[a] for a in kept_auth]
        for a_id in ids:
            by_year_author_papers[year][a_id] += 1
        # all unordered pairs become symmetric directed edges
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                s, d = ids[i], ids[j]
                by_year_pair[year][(s, d)] += 1.0
                by_year_pair[year][(d, s)] += 1.0

    graphs = []
    edge_count_per_snapshot = []
    active_count_per_snapshot = []
    years_kept = []
    for year in sorted(by_year_pair.keys()):
        pairs = by_year_pair[year]

        active = set()
        src_list = []
        dst_list = []
        weight_list = []
        for (s, d), w in pairs.items():
            src_list.append(s)
            dst_list.append(d)
            weight_list.append(w)
            active.add(s)
            active.add(d)

        if len(active) < min_active_nodes:
            continue

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        weight_tensor = torch.tensor(weight_list, dtype=torch.float)

        # per-node total publication count for the year (in kept_set)
        node_papers = torch.zeros(n_nodes)
        for a_id, c in by_year_author_papers[year].items():
            node_papers[a_id] = float(c)
        papers_max = node_papers.max().clamp(min=1e-8)
        node_papers_norm = (node_papers / papers_max).unsqueeze(1)  # [N, 1]

        structural = compute_structural_features(edge_index, n_nodes=n_nodes, edge_weights=weight_tensor)
        x = torch.cat([node_papers_norm, structural], dim=1)  # [N, 6]

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=weight_tensor.unsqueeze(1),
            node_ids=torch.arange(n_nodes),
        ))
        edge_count_per_snapshot.append(edge_index.shape[1])
        active_count_per_snapshot.append(len(active))
        years_kept.append(year)

    n = len(graphs)
    train_range, val_range, test_range = compute_split_ranges(n)
    meta = {
        "dataset": "dblp",
        "n_nodes": n_nodes,
        "n_snapshots": n,
        "node_feature_dim": 6,
        "min_papers": min_papers,
        "max_authors": max_authors,
        "years_kept": years_kept,
        "edge_counts": edge_count_per_snapshot,
        "active_counts": active_count_per_snapshot,
        "train_range": list(train_range),
        "val_range": list(val_range),
        "test_range": list(test_range),
    }
    return graphs, meta


def build_dblp_graphs(
    xml_gz_path: str,
    year_min: int = 1995,
    year_max: int = 2020,
    min_papers: int = 5,
    min_active_nodes: int = 30,
    max_authors: int = 1000,
):
    """full pipeline: parse dblp xml, filter authors, build annual snapshots.

    args:
        xml_gz_path: path to dblp.xml.gz
        year_min, year_max: year window
        min_papers: drop authors with fewer than this many papers across window
        min_active_nodes: drop years below this active author count
        max_authors: cap on author set size (top by paper count)
    """
    print(f"streaming dblp xml from {xml_gz_path}...")
    records, counts = _collect_paper_records(xml_gz_path, year_min, year_max)
    print(f"collected {len(records)} papers across {len(counts)} unique authors")
    return build_dblp_graphs_from_records(
        records,
        counts,
        min_papers=min_papers,
        min_active_nodes=min_active_nodes,
        max_authors=max_authors,
    )
