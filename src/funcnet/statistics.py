"""Small statistical summaries shared by analysis tutorials.

Only general, explicitly parameterized statistics belong here.  Dataset-specific
pooling (for example, how repeated recording days are nested within a mouse)
stays in the tutorial that defines that scientific weighting choice.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats


def mean_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    axis: int | tuple[int, ...] | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a NaN-aware Student-*t* confidence interval for the mean.

    Parameters
    ----------
    values
        Observations.  Missing values may be represented by ``NaN``.
    confidence
        Central confidence level in ``(0, 1)``; the default is 95%.
    axis
        Observation axis (or axes).  The default treats rows as observations
        and computes one interval per column.  ``None`` summarizes all values.

    Returns
    -------
    mean, lower, upper
        Arrays with the selected axes removed.  Bounds are ``NaN`` wherever
        fewer than two valid observations are available.

    Notes
    -----
    The interval uses the sample standard deviation (``ddof=1``) and a
    per-output valid count, so columns with different amounts of missing data
    receive the correct degrees of freedom.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    values = np.asarray(values, dtype=float)
    with warnings.catch_warnings():
        # All-NaN slices and slices with one observation are represented by NaN
        # below; their NumPy warnings do not add useful information to callers.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=axis)
        sample_sd = np.nanstd(values, axis=axis, ddof=1)

    n_valid = np.sum(~np.isnan(values), axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        sem = sample_sd / np.sqrt(n_valid)
        critical = stats.t.ppf((1.0 + confidence) / 2.0, n_valid - 1)
        margin = critical * sem

    valid = n_valid >= 2
    lower = np.where(valid, mean - margin, np.nan)
    upper = np.where(valid, mean + margin, np.nan)
    return mean, lower, upper


__all__ = ["mean_confidence_interval"]
