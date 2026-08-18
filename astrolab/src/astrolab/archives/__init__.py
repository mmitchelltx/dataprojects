"""Archive clients.

Science modules never import from here directly -- they consume calibrated data products.
This layer's job is retrieval, caching, and recording what was retrieved.

Every client in this package obeys three rules:

1. A query that returns nothing raises :class:`~astrolab.archives.base.EmptyResultError`.
   It never returns an empty structure that a caller might mistake for a null result, and it
   never substitutes anything.
2. Every query is expressed as a :class:`~astrolab.archives.base.QuerySpec`, which is hashable,
   serialisable, and sufficient to replay the retrieval.
3. Every retrieval produces a :class:`~astrolab.core.provenance.QueryRecord` stating whether
   the answer came from cache or from the live archive.
"""
