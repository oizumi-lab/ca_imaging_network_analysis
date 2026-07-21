# %% [markdown]
# # 02 · Report-ready full-neuron contrasts
#
# This lightweight final stage converts the exact output tables into the
# comparisons needed to interpret estimator effects: alpha-path sparsity,
# awake-to-unconscious contrasts, method-by-state interactions, and agreement
# between Louvain partitions.

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

SCRIPT_PATH = Path(__file__).resolve()
GLASSO_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(GLASSO_ROOT / "src"))

from glasso_analysis.config import COMPARISON_DIR, CONFIG, MATRIX_DIR  # noqa: E402

MEASURES = [
    "modularity_Q",
    "n_modules",
    "clustering",
    "path_length_giant",
    "giant_component_fraction",
    "degree_cv",
    "n_isolates",
    "edge_distance_mean_um",
]

# %% [markdown]
# ## Alpha-path sparsity and convergence table

# %%
alpha_rows: list[dict] = []
for recording_name, settings in CONFIG["recordings"].items():
    for state in settings["states"]:
        path = MATRIX_DIR / recording_name / state / "alpha_diagnostics.json"
        for item in json.loads(path.read_text(encoding="utf-8")):
            alpha_rows.append({"recording": recording_name, "state": state, **item})

alpha_table = pd.DataFrame(alpha_rows).sort_values(
    ["recording", "state", "alpha"], ascending=[True, True, False]
)
alpha_table.to_csv(COMPARISON_DIR / "alpha_path.csv", index=False)

# %% [markdown]
# ## Awake-to-unconscious state contrasts and estimator interactions

# %%
network = pd.read_csv(COMPARISON_DIR / "network_measures.csv")
available = network[network["available"]].copy()
contrast_rows: list[dict] = []
for recording_name, settings in CONFIG["recordings"].items():
    awake, unconscious = settings["states"]
    recording_rows = available[available["recording"] == recording_name]
    for (method, density), group in recording_rows.groupby(
        ["method", "requested_density"]
    ):
        by_state = group.set_index("state")
        if awake not in by_state.index or unconscious not in by_state.index:
            continue
        row = {
            "recording": recording_name,
            "awake_state": awake,
            "unconscious_state": unconscious,
            "method": method,
            "density": density,
        }
        for measure in MEASURES:
            row[f"awake_{measure}"] = float(by_state.loc[awake, measure])
            row[f"unconscious_{measure}"] = float(by_state.loc[unconscious, measure])
            row[f"delta_{measure}"] = (
                row[f"unconscious_{measure}"] - row[f"awake_{measure}"]
            )
        contrast_rows.append(row)

contrast_table = pd.DataFrame(contrast_rows).sort_values(
    ["recording", "density", "method"]
)
contrast_table.to_csv(COMPARISON_DIR / "state_contrasts.csv", index=False)

interaction_rows: list[dict] = []
for (recording_name, density), group in contrast_table.groupby(["recording", "density"]):
    by_method = group.set_index("method")
    if "pearson" not in by_method.index:
        continue
    for method in ("glasso_marginal", "glasso_partial"):
        if method not in by_method.index:
            continue
        row = {
            "recording": recording_name,
            "density": density,
            "method": method,
        }
        for measure in MEASURES:
            row[f"interaction_{measure}"] = float(
                by_method.loc[method, f"delta_{measure}"]
                - by_method.loc["pearson", f"delta_{measure}"]
            )
        interaction_rows.append(row)

interaction_table = pd.DataFrame(interaction_rows).sort_values(
    ["recording", "density", "method"]
)
interaction_table.to_csv(COMPARISON_DIR / "method_state_interactions.csv", index=False)

# %% [markdown]
# ## Partition agreement at matched state and density

# %%
partition_rows: list[dict] = []
for recording_name, settings in CONFIG["recordings"].items():
    for state in settings["states"]:
        for density in CONFIG["network_densities"]:
            reference_path = (
                COMPARISON_DIR
                / f"partition_{recording_name}_{state}_pearson_K{density:.4f}.npy"
            )
            if not reference_path.exists():
                continue
            reference = np.load(reference_path)
            for method in ("glasso_marginal", "glasso_partial"):
                method_path = (
                    COMPARISON_DIR
                    / f"partition_{recording_name}_{state}_{method}_K{density:.4f}.npy"
                )
                if not method_path.exists():
                    continue
                partition = np.load(method_path)
                partition_rows.append(
                    {
                        "recording": recording_name,
                        "state": state,
                        "density": density,
                        "comparison": f"pearson_vs_{method}",
                        "adjusted_rand_index": float(
                            adjusted_rand_score(reference, partition)
                        ),
                    }
                )

partition_table = pd.DataFrame(partition_rows).sort_values(
    ["recording", "state", "density", "comparison"]
)
partition_table.to_csv(COMPARISON_DIR / "partition_agreement.csv", index=False)

# %% [markdown]
# ## Console audit

# %%
print("alpha path rows:", len(alpha_table))
print("state contrasts:", len(contrast_table))
print("method-state interactions:", len(interaction_table))
print("partition comparisons:", len(partition_table))
print("saved report-ready tables ->", COMPARISON_DIR)

