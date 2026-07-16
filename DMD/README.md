# DMD research — private

This directory contains the dynamic mode decomposition (DMD) research project. It is intentionally separate from the Kiyooka et al. reproduction and summer-school materials at the repository root.

The initial-validation smoke test and the focused PyDMD window-tuning audit are
implemented as private, staged analyses. They include the literature survey,
preregistration-style plans, configuration, reusable code, tests, step-by-step
interactive scripts, immutable result artifacts, and detailed reports. The
current verification deliberately stops before all-recording brain-state or
Grassmann-trajectory analysis because its prediction and scientific mode gates
did not pass. Against one fixed reference shared by every forecast target, all
143 guarded configurations have negative predictive R². The lowest common
loss occurs at 180 s, whereas the largest window-specific local-mean increment
occurs at 45 s and is only 0.0089. Neither window is verified, and no
low-overlap recurrent mode track is established; these are diagnostics, not a
parameter recommendation for production sliding-window analysis.

## Directory boundary

- `documents/` — DMD plans, reports, LaTeX sources, and bibliography.
- `references/` — DMD-specific external papers. PDFs are kept locally and ignored by Git.
- `configs/` — frozen machine-readable pilot protocol.
- `scripts/` — command-line entry points and tutorial-style interactive `# %%`
  analysis scripts.
- `src/dmd_validation/` — selective data access, preprocessing, DMD, validation,
  bootstrap, and null-control implementation.
- `tests/` — focused numerical and boundary-safety regression tests.
- `results/` — generated DMD outputs. Contents are ignored by Git except the directory documentation.

New DMD code should not be placed in the repository-root `src/` or `scripts/`
trees.

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
