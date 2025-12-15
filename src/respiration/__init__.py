"""
Respiration rate estimation module for VitaStress project.

This module provides tools for extracting respiratory rate from PPG signals
using the RRest toolbox (Charlton et al. 2016) via MATLAB Engine API.
"""

from .rrest_vitastress import (
    check_ppg_missing_values,
    load_ppg_from_csv,
    find_subject_ppg_files,
    prepare_rrest_data,
    configure_rrest_params,
    run_rrest_analysis,
)

__all__ = [
    "check_ppg_missing_values",
    "load_ppg_from_csv",
    "find_subject_ppg_files",
    "prepare_rrest_data",
    "configure_rrest_params",
    "run_rrest_analysis",
]

