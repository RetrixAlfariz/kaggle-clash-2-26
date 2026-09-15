"""Deterministic raw-text span features for train-only structure diagnostics.

The extractor has no label, model, or dataset dependencies.  Token bounds use
the half-open character offsets returned by ``sequence_crf.tokens``.
"""
from __future__ import annotations

from bisect import bisect_right
import math
import re

import numpy as np


# Kept explicit and ordered so diagnostic matrices have a stable schema.
CONTROL_NAMES = [
    "token_width_log1p", "char_width_log1p", "span_upper_fraction",
    "span_digit_fraction", "span_punctuation_fraction",
    "first_title", "first_upper", "first_digit",
    "last_title", "last_upper", "last_digit",
    "newline_before_first", "newline_after_first",
    "newline_before_last", "newline_after_last",
    "newline_before_mean", "newline_after_mean",
    "adjacent_previous_first", "adjacent_next_first",
    "adjacent_previous_last", "adjacent_next_last",
]

STRUCTURE_NAMES = [
    "line_position_start", "line_position_end",
    "paragraph_index_start", "paragraph_index_end",
    "starts_paragraph", "ends_paragraph", "crosses_line",
    "preceding_colon_same_line", "following_colon_same_line",
    "distance_from_preceding_colon", "colon_within_span",
    "document_char_position_start", "document_char_position_end",
    "previous_physical_line_nonblank", "next_physical_line_nonblank",
    "previous_nonblank_line_has_colon", "next_nonblank_line_has_colon",
    "distance_to_previous_nonblank_line", "distance_to_next_nonblank_line",
]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _unit(value: float) -> float:
    """Finite clipping for normalized positions/distances."""
    return min(1.0, max(0.0, float(value)))


class DocumentLayout:
    """Precompute line and paragraph layout for one document."""

    def __init__(self, text: str, bounds: np.ndarray):
        self.text = text
        self.bounds = np.asarray(bounds, dtype=np.int64)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (N, 2)")
        if len(self.bounds):
            if np.any(self.bounds[:, 0] < 0) or np.any(self.bounds[:, 1] <= self.bounds[:, 0]) or np.any(self.bounds[:, 1] > len(text)):
                raise ValueError("token bounds must be valid half-open character offsets")
            if np.any(self.bounds[1:, 0] < self.bounds[:-1, 1]):
                raise ValueError("token bounds must be ordered and non-overlapping")

        # splitlines(keepends=True) preserves CRLF offsets and represents the
        # final unterminated line. Empty text is a single empty physical line.
        self.lines = text.splitlines(keepends=True) or [""]
        self.line_starts = [0]
        for line in self.lines[:-1]:
            self.line_starts.append(self.line_starts[-1] + len(line))
        self.line_ends = [s + len(line) for s, line in zip(self.line_starts, self.lines)]
        self.line_content = [line.rstrip("\r\n") for line in self.lines]
        self.line_nonblank = [bool(line.strip()) for line in self.line_content]
        self.line_colon = [":" in line for line in self.line_content]

        self.token_lines = np.asarray(
            [self._line_for_char(int(a)) for a, _ in self.bounds], dtype=np.int32
        )
        # A blank physical line starts a new paragraph. Consecutive blank
        # lines do not create empty paragraph IDs.
        paragraph = 0
        self.line_paragraph = []
        seen_content = False
        pending_break = False
        for nonblank in self.line_nonblank:
            if nonblank:
                if seen_content and pending_break:
                    paragraph += 1
                seen_content = True
                pending_break = False
            elif seen_content:
                pending_break = True
            self.line_paragraph.append(paragraph)
        # Paragraph boundaries are defined by first/last token-bearing line,
        # avoiding blank/header-only lines changing token membership.
        token_lines = self.token_lines.tolist()
        self.token_paragraph = np.asarray(
            [self.line_paragraph[line] for line in token_lines], dtype=np.int32
        )
        self.paragraph_token_first: dict[int, int] = {}
        self.paragraph_token_last: dict[int, int] = {}
        for i, p in enumerate(self.token_paragraph.tolist()):
            self.paragraph_token_first.setdefault(p, i)
            self.paragraph_token_last[p] = i
        self._nonblank_before = self._nearest_nonblank(previous=True)
        self._nonblank_after = self._nearest_nonblank(previous=False)
        self._colon_positions = [m.start() for m in re.finditer(":", text)]
        self._colon_by_line: dict[int, list[int]] = {}
        for pos in self._colon_positions:
            self._colon_by_line.setdefault(self._line_for_char(pos), []).append(pos)

    def _line_for_char(self, pos: int) -> int:
        if not self.lines:
            return 0
        return min(len(self.lines) - 1, max(0, bisect_right(self.line_starts, pos) - 1))

    def _nearest_nonblank(self, *, previous: bool) -> list[int]:
        result = []
        last = -1
        indices = range(len(self.lines)) if previous else range(len(self.lines) - 1, -1, -1)
        lookup = [-1] * len(self.lines)
        for i in indices:
            lookup[i] = last
            if self.line_nonblank[i]:
                last = i
        # For following lookup, reverse pass lookup is indexed correctly.
        return lookup

    def features(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (control, structure) float64 features for token interval [a,b)."""
        n = len(self.bounds)
        if not (0 <= a < b <= n):
            raise ValueError("token interval must satisfy 0 <= a < b <= len(bounds)")
        start, end = map(int, (self.bounds[a, 0], self.bounds[b - 1, 1]))
        raw = self.text[start:end]
        chars = list(raw)
        denom = max(1, len(chars))
        words = [self.text[int(x):int(y)] for x, y in self.bounds[a:b]]
        first, last = words[0], words[-1]
        token_count = b - a
        char_width = end - start

        def newline_before(i: int) -> float:
            left = int(self.bounds[i - 1, 1]) if i else 0
            return float("\n" in self.text[left:int(self.bounds[i, 0])])

        def newline_after(i: int) -> float:
            right = int(self.bounds[i + 1, 0]) if i + 1 < n else len(self.text)
            return float("\n" in self.text[int(self.bounds[i, 1]):right])

        before = [newline_before(i) for i in range(a, b)]
        after = [newline_after(i) for i in range(a, b)]
        punct = sum(1 for c in chars if not c.isalnum() and not c.isspace() and c != "_")
        control = np.asarray([
            math.log1p(token_count), math.log1p(char_width),
            sum(c.isupper() for c in chars) / denom,
            sum(c.isdigit() for c in chars) / denom,
            punct / denom,
            float(first.istitle()), float(first.isupper()), float(first.isdigit()),
            float(last.istitle()), float(last.isupper()), float(last.isdigit()),
            newline_before(a), newline_after(a), newline_before(b - 1), newline_after(b - 1),
            float(np.mean(before)), float(np.mean(after)),
            float(a > 0 and int(self.bounds[a - 1, 1]) == int(self.bounds[a, 0])),
            float(a + 1 < n and int(self.bounds[a, 1]) == int(self.bounds[a + 1, 0])),
            float(b - 1 > 0 and int(self.bounds[b - 2, 1]) == int(self.bounds[b - 1, 0])),
            float(b < n and int(self.bounds[b - 1, 1]) == int(self.bounds[b, 0])),
        ], dtype=np.float64)

        start_line, end_line = int(self.token_lines[a]), int(self.token_lines[b - 1])
        line_den = max(1, len(self.lines) - 1)
        start_para, end_para = int(self.token_paragraph[a]), int(self.token_paragraph[b - 1])
        para_den = max(1, max(self.line_paragraph, default=0))
        start_line_text = self.line_content[start_line]
        end_line_text = self.line_content[end_line]
        # Colon relation is local to the span's boundary line. Distances are
        # character distances divided by document length and clipped to [0,1].
        start_colons = self._colon_by_line.get(start_line, [])
        end_colons = self._colon_by_line.get(end_line, [])
        before_i = bisect_right(start_colons, start - 1) - 1
        after_i = bisect_right(end_colons, end - 1)
        nearest_before = start_colons[before_i] if before_i >= 0 else None
        has_after = after_i < len(end_colons)
        prev_line = self._nonblank_before[start_line]
        next_line = self._nonblank_after[end_line]
        # End offsets may equal len(text), so use document length as divisor.
        doc_den = max(1, len(self.text))
        structure = np.asarray([
            start_line / line_den, end_line / line_den,
            start_para / para_den, end_para / para_den,
            float(self.paragraph_token_first.get(start_para) == a),
            float(self.paragraph_token_last.get(end_para) == b - 1),
            float(start_line != end_line),
            float(nearest_before is not None), float(has_after),
            _unit((start - nearest_before) / max(1, len(self.text))) if nearest_before is not None else 1.0,
            float(":" in raw),
            start / doc_den, end / doc_den,
            float(start_line > 0 and self.line_nonblank[start_line - 1]),
            float(end_line + 1 < len(self.lines) and self.line_nonblank[end_line + 1]),
            float(prev_line >= 0 and self.line_colon[prev_line]),
            float(next_line >= 0 and self.line_colon[next_line]),
            (start_line - prev_line) / line_den if prev_line >= 0 else 1.0,
            (next_line - end_line) / line_den if next_line >= 0 else 1.0,
        ], dtype=np.float64)
        if not np.isfinite(control).all() or not np.isfinite(structure).all():
            raise AssertionError("feature extractor produced non-finite values")
        return control, structure


def features(text: str, bounds: np.ndarray, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Convenience API; construct once per document for repeated span queries."""
    return DocumentLayout(text, bounds).features(a, b)
