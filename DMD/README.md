# DMD research — private

This directory contains the dynamic mode decomposition (DMD) research project. It is intentionally separate from the Kiyooka et al. reproduction and summer-school materials at the repository root.

No DMD analysis has been implemented yet. The current contents are the literature survey, preregistration-style analysis plan, and the two DMD-specific source papers.

## Directory boundary

- `documents/` — DMD plans, reports, LaTeX sources, and bibliography.
- `references/` — DMD-specific external papers. PDFs are kept locally and ignored by Git.
- `scripts/` — future DMD command-line entry points.
- `results/` — generated DMD outputs. Contents are ignored by Git except the directory documentation.

Add `src/`, `tests/`, and `configs/` inside this directory when implementation begins. New DMD code should not be placed in the repository-root `src/` or `scripts/` trees.

## Shared inputs during the transition

For now, DMD work may read the existing calcium-imaging dataset from `../data/` and use the Kiyooka et al. paper in `../raw_documents/` as scientific context. Treat these as external, read-only dependencies. Use configured paths rather than hard-coded absolute paths so this directory can later move to its own repository.

If root-level Kiyooka analysis logic must be reused, expose it through a narrow, documented adapter. Before repository separation, either move that adapter into the DMD project or package genuinely shared code explicitly; avoid deep imports across the directory boundary.

## Student-release privacy

This folder boundary is an organizational safeguard, not an access-control boundary. If this combined Git repository or its `.git` history is shared, DMD files that were committed can remain recoverable even after the directory is deleted.

Before distributing summer-school material:

1. Create the student repository/export from an explicit allowlist of Kiyooka-related files.
2. Put this `DMD/` tree in a separate private repository or private remote.
3. Do not distribute the combined repository's `.git` directory or any branch containing DMD commits.
4. Verify the student export from a fresh clone or archive before release.

Separating the repositories before DMD implementation starts is the safest option and minimizes later dependency cleanup.
