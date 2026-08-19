"""Variable stars and time-domain analysis.

Shares the time-series core with the exoplanet pillar -- light curves, folding, periodograms --
but the problem is different in a way that shapes the code. A transit search knows what it is
looking for and can use a matched template; a variable-star search has to determine a period
from sparse, irregularly sampled ground-based data where the sampling itself imprints strong
false signals. Alias handling is therefore not an afterthought here, it is the main event.
"""
