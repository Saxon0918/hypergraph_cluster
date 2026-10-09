import gzip

import pytest

from hce.data.hic import parse_hic
from hce.data.scholp import parse_scholp
from hce.exceptions import DataFormatError


def test_hic_edge_first_token_is_a_member_not_size(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text("1\n4 2 7\nA B C D\n3 1\n0 2 3\n")
    sample = parse_hic(path, "tiny")[0]
    assert sample.hyperedges == ((3, 1), (0, 2, 3))
    assert sample.graph_label == 7


def test_hic_does_not_guess_one_based_indexing(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1\n3 1 0\nA B C\n1 3\n")
    with pytest.raises(DataFormatError):
        parse_hic(path, "tiny", index_base=0)
    assert parse_hic(path, "tiny", index_base=1)[0].hyperedges == ((0, 2),)


def _write_gzip(path, values):
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(map(str, values)))


def test_scholp_offsets_and_stable_time_sort(tmp_path):
    _write_gzip(tmp_path / "x-simplices.txt.gz", [10, 20, 30, 40, 50])
    _write_gzip(tmp_path / "x-nverts.txt.gz", [2, 1, 2])
    _write_gzip(tmp_path / "x-times.txt.gz", [8, 3, 8])
    table = parse_scholp(tmp_path, "x", "days")
    assert table.events == ((30,), (10, 20), (40, 50))
    assert table.times == (3, 8, 8)
    assert table.original_event_indices == (1, 0, 2)


def test_scholp_length_mismatch_is_failure(tmp_path):
    (tmp_path / "x-simplices.txt").write_text("1\n2\n")
    (tmp_path / "x-nverts.txt").write_text("3\n")
    (tmp_path / "x-times.txt").write_text("9\n")
    with pytest.raises(DataFormatError):
        parse_scholp(tmp_path, "x", "days")

