KAGGLE CLASH 2 - IRIS 2026
Sensitive Information Extraction from Business Correspondence
================================================================================

69,468 labelled training documents / 23,156 test documents / 20,000 extra
weakly-labelled documents. Entity labels: NAME, DATE, EMAIL, PHONE, JOB_TITLE, ADDRESS, USERNAME

--------------------------------------------------------------------------------
1. HOW TO REBUILD A DOCUMENT
--------------------------------------------------------------------------------
No file contains a whole document. Text is stored one line per row in
train_segments.csv / test_segments.csv, and THE ROWS ARE SHUFFLED.

To rebuild the exact text of a document:

    1. take every row with that document_id
    2. sort them by segment_index ascending
    3. join the `text` values with a single newline character "\n"

    full_text = "\n".join(group.sort_values("segment_index")["text"])

Two things break this if you are not careful:

    * Blank lines are real rows with an empty `text` value. They are part of the
      document and must be kept. If a blank line goes missing during
      reconstruction, every character offset after that point is wrong.
    * Rows are shuffled on purpose. Do not rely on file order.

Check yourself: `n_chars` in the metadata file is the true length of the rebuilt
document. If len(full_text) != n_chars, your reconstruction is wrong.

All character offsets in this competition - in train_labels.csv and in your
submission - refer to that rebuilt string, counted in Python characters from 0,
with end exclusive: full_text[start_offset:end_offset] is the entity text.

--------------------------------------------------------------------------------
2. FILES
--------------------------------------------------------------------------------
train/train_segments.csv
    document_id     document this line belongs to
    segment_index   0-based position of the line inside the document
    text            the line itself (may be empty)

train/train_labels.csv          (shuffled; join to the text on document_id)
    document_id, label, start_offset, end_offset

train/train_metadata.parquet    (Apache Parquet, one row per document)
    document_id
    channel               EMAIL / MEMO / LETTER      derived from the text
    domain                LEGAL / HEALTHCARE / FINANCE / HR / IT / GENERAL
                                                     derived from the text
    n_chars               length of the rebuilt document  (use this to verify)
    n_segments            number of lines
    n_words               whitespace-delimited word count
    n_nonempty_segments   lines with non-whitespace content
    source_system         synthetic operational field, no predictive value
    ingested_at           synthetic timestamp, no predictive value

test/test_segments.csv          same schema as train_segments.csv
test/test_metadata.parquet      same schema as train_metadata.parquet

extra/weak_labeled_documents.jsonl
    20,000 additional documents with full_text included directly, annotated by
    an automatic regex system rather than by a human. Field `weak_entities`.
    These labels are NOISY AND INCOMPLETE BY DESIGN, and they do not cover all
    seven categories equally. Measuring their quality for yourself is part of the
    work. Using this file is optional. Deciding whether and how to use it is part
    of the competition.

sample_submission.csv           the exact rows your submission must contain

--------------------------------------------------------------------------------
3. SUBMISSION
--------------------------------------------------------------------------------
Exactly two columns, 285,318 rows, and the row_id values and their order must
match sample_submission.csv exactly.

    row_id      "<document_id>_<NN>", NN starting at 01
    Predicted   "<LABEL>:<start_offset>:<end_offset>",  e.g.  NAME:84:97

The number of slots for a document is the number of entities that document
actually contains. Slot NN holds the NN-th entity of the document ORDERED BY
start_offset ASCENDING (ties broken by end_offset ascending). Sort your
predicted spans that way before writing them out.

A row scores 1 only if the label and both offsets are exactly right.
