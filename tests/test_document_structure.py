import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_structure import CONTROL_NAMES, STRUCTURE_NAMES, DocumentLayout, features
from neural_sequence import encode_document
from sequence_crf import tokens


class DocumentStructureTests(unittest.TestCase):
  def test_schema_and_control_newline_adjacency_match_encoder_flags(self):
    text = "A B\nC: D"
    ts = tokens(text)
    bounds = np.asarray([[a, b] for a, b, _ in ts], dtype=np.int32)
    _, _, _, flags = encode_document(text, {})
    layout = DocumentLayout(text, bounds)
    control, structure = layout.features(1, 4)
    self.assertEqual(control.dtype, structure.dtype)
    self.assertEqual(control.dtype, np.float64)
    self.assertEqual(len(control), len(CONTROL_NAMES))
    self.assertEqual(len(structure), len(STRUCTURE_NAMES))
    self.assertEqual(control[CONTROL_NAMES.index("newline_before_first")], flags[1, 3])
    self.assertEqual(control[CONTROL_NAMES.index("newline_after_first")], flags[1, 4])
    self.assertEqual(control[CONTROL_NAMES.index("newline_before_last")], flags[3, 3])
    self.assertEqual(control[CONTROL_NAMES.index("newline_after_last")], flags[3, 4])
    self.assertEqual(control[CONTROL_NAMES.index("adjacent_previous_first")], flags[1, 5])
    self.assertEqual(control[CONTROL_NAMES.index("adjacent_next_last")], flags[3, 6])
    self.assertEqual(structure[STRUCTURE_NAMES.index("crosses_line")], 1)
    self.assertTrue(np.isfinite(control).all() and np.isfinite(structure).all())


  def test_paragraph_indices_boundaries_and_blank_line_handling_crlf(self):
    text = "Header: x\r\nline two\r\n\r\nBody\r\nlast"
    ts = tokens(text)
    bounds = np.asarray([[a, b] for a, b, _ in ts], dtype=np.int32)
    layout = DocumentLayout(text, bounds)
    header_i = next(i for i, (_, _, w) in enumerate(ts) if w == "Header")
    line_two_i = next(i for i, (_, _, w) in enumerate(ts) if w == "line")
    line_two_last_i = next(i for i, (_, _, w) in enumerate(ts) if w == "two")
    body_i = next(i for i, (_, _, w) in enumerate(ts) if w == "Body")
    last_i = next(i for i, (_, _, w) in enumerate(ts) if w == "last")
    self.assertEqual(layout.features(header_i, line_two_i + 1)[1][STRUCTURE_NAMES.index("starts_paragraph")], 1)
    self.assertEqual(layout.features(header_i, line_two_last_i + 1)[1][STRUCTURE_NAMES.index("ends_paragraph")], 1)
    body_features = layout.features(body_i, body_i + 1)[1]
    self.assertEqual(body_features[STRUCTURE_NAMES.index("paragraph_index_start")], 1)
    self.assertEqual(body_features[STRUCTURE_NAMES.index("starts_paragraph")], 1)
    self.assertEqual(layout.features(last_i, last_i + 1)[1][STRUCTURE_NAMES.index("paragraph_index_end")], 1)


  def test_colon_relations_and_nonblank_line_adjacency(self):
    text = "To: Alex\n\nName\nRole: Analyst\n"
    ts = tokens(text)
    bounds = np.asarray([[a, b] for a, b, _ in ts], dtype=np.int32)
    layout = DocumentLayout(text, bounds)
    alex = next(i for i, (_, _, w) in enumerate(ts) if w == "Alex")
    name = next(i for i, (_, _, w) in enumerate(ts) if w == "Name")
    role = next(i for i, (_, _, w) in enumerate(ts) if w == "Role")
    _, alex_struct = layout.features(alex, alex + 1)
    _, name_struct = layout.features(name, name + 1)
    _, role_struct = layout.features(role, role + 1)
    self.assertEqual(alex_struct[STRUCTURE_NAMES.index("preceding_colon_same_line")], 1)
    self.assertLess(alex_struct[STRUCTURE_NAMES.index("distance_from_preceding_colon")], 1)
    self.assertEqual(name_struct[STRUCTURE_NAMES.index("previous_physical_line_nonblank")], 0)
    self.assertEqual(name_struct[STRUCTURE_NAMES.index("previous_nonblank_line_has_colon")], 1)
    self.assertEqual(role_struct[STRUCTURE_NAMES.index("next_physical_line_nonblank")], 0)
    self.assertEqual(role_struct[STRUCTURE_NAMES.index("colon_within_span")], 0)
    colon_i = next(i for i, (_, _, w) in enumerate(ts) if w == ":")
    self.assertEqual(layout.features(colon_i, colon_i + 1)[1][STRUCTURE_NAMES.index("colon_within_span")], 1)


  def test_single_line_and_document_edges_are_finite_and_normalized(self):
    text = "One\r\n\r\nTwo"
    ts = tokens(text)
    bounds = np.asarray([[a, b] for a, b, _ in ts], dtype=np.int32)
    layout = DocumentLayout(text, bounds)
    first, last = layout.features(0, 1)[1], layout.features(1, 2)[1]
    self.assertEqual(first[STRUCTURE_NAMES.index("line_position_start")], 0)
    self.assertEqual(last[STRUCTURE_NAMES.index("line_position_start")], 1)
    self.assertTrue(np.all((first >= 0) & (first <= 1)))
    self.assertTrue(np.all((last >= 0) & (last <= 1)))
    empty_layout = DocumentLayout("", np.empty((0, 2), dtype=np.int32))
    self.assertEqual(len(empty_layout.lines), 1)


  def test_invalid_bounds_rejected(self):
    for bounds in (np.asarray([[0, 2], [1, 3]]), np.asarray([[-1, 1]]), np.asarray([[0, 99]])):
      with self.assertRaises(ValueError):
        DocumentLayout("abc", bounds)


  def test_invalid_span_interval_rejected_and_convenience_api(self):
    text = "A: B"
    bounds = np.asarray([[a, b] for a, b, _ in tokens(text)], dtype=np.int32)
    expected = DocumentLayout(text, bounds).features(0, 2)
    actual = features(text, bounds, 0, 2)
    np.testing.assert_array_equal(expected[0], actual[0])
    np.testing.assert_array_equal(expected[1], actual[1])
    with self.assertRaises(ValueError):
        DocumentLayout(text, bounds).features(1, 1)
