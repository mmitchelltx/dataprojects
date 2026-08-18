"""Per-instrument calibration quirks and ingestion.

Everything mission-specific lives here: time systems, cadence conventions, quality-flag
vocabularies, known systematics. Science modules consume the calibrated
:class:`~astrolab.core.lightcurve.LightCurve` products this layer produces and never learn
which instrument they came from.

That boundary is what lets the same transit search run on K2 and TESS unchanged, and what
keeps a K2-specific correction from silently being applied to data it does not describe.
"""
