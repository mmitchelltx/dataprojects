# Architecture Decision Records

One file per decision that was not obvious. Each records the context, the options considered, the
choice, and — most importantly — the *consequences we accepted*.

The rule for writing one: if a future reader could reasonably ask "why on earth did they do it that
way," there should be an ADR answering it. Domain decisions (which detrending method, which prior,
which reference file) count as much as software ones, and get explained at a level that assumes
strong Python and *no* prior instrument-systematics knowledge.

ADRs are immutable once accepted. A reversed decision gets a new ADR that supersedes the old one;
the old one stays, marked superseded. The history of what was believed when is part of the record.

| # | Title | Status |
|---|---|---|
| [0001](0001-repo-layout-and-environment.md) | Repository layout and environment manager | Accepted |
| [0002](0002-uncertainty-representation.md) | Posterior samples as the canonical uncertainty type | Accepted |
| [0003](0003-two-stage-detrending.md) | Two-stage detrending: filter to search, joint-fit to measure | Proposed |
| [0004](0004-real-pinned-data-in-ci.md) | CI runs on pinned real observations, never synthesized fixtures | Accepted |
| [0005](0005-dependency-selection.md) | Dependency selection and maintenance posture | Accepted |
