# Related methodology

[Ghosh et al. (2022), Span Classification with Structured Information for Disfluency Detection in Spoken Utterances](2203.16028v2.pdf), arXiv2203.16028v2. Archived byte-for-byte from the PDF supplied by Vian; source location was `C:/Users/Revian/Downloads/Documents/2203.16028v2.pdf`.

Role: related methodology, not transferable performance evidence. The paper uses pretrained contextual encoders and syntactic dependency graphs for a different task. Its span representation and task-relevant structure motivated a testable document-structure hypothesis for this project; its reported gains do not predict our gains.

The [train-only diagnostic](../bitrase-2/structure-diagnostic/report.md) found small residual candidate-level information beyond P0 scores, shape, and existing newline/adjacency controls, but no practically useful hard-pair ordering improvement. The predefined structural-feature scorer branch is stopped. This does not establish that all document structure is useless, nor that structure cannot help unseen templates: the selected grouping proxies were all singletons, and P0 had already trained on those documents.

No line embeddings, signature-role module, graph/GCN, or additional structural feature search is justified by this result. The next open investigation is posterior concentration in the frozen P1 scoring/decoding system, tracked in the [bitrase-2 index](../bitrase-2/README.md).
