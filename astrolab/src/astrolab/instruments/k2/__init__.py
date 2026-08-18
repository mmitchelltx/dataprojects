"""K2 ingestion and instrument conventions."""

from astrolab.instruments.k2.ingest import (
    K2_CAMPAIGN_1_SPAN,
    THIRD_PARTY_MIRROR,
    load_k2_csv,
    load_validation_lightcurve,
)

__all__ = [
    "K2_CAMPAIGN_1_SPAN",
    "THIRD_PARTY_MIRROR",
    "load_k2_csv",
    "load_validation_lightcurve",
]
