# DMD documents

- `01_dmd_tracking_literature_survey.pdf` — critical review of Raut, Germain, and related methods.
- `02_dmd_brain_state_analysis_plan.pdf` — preregistration-style research and validation plan.
- Matching `.tex` files are the editable sources.
- `dmd_tracking_references.bib` is the shared bibliography.

Build from this directory:

```bash
latexmk -pdf 01_dmd_tracking_literature_survey.tex
latexmk -pdf 02_dmd_brain_state_analysis_plan.tex
```
