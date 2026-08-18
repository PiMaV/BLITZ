# Auxiliary content

Benchmarks, external posts, and tests. Kept in one folder to keep the BLITZ
repo root minimal.

- **benchmarks/** – performance and load benchmarks
- **benchmarks/sparse_sidecar/** – PLAN D mini ImageView (not the BLITZ app).
  Two viewers side by side + scrub numbers, or
  `uv run python _aux/benchmarks/sparse_sidecar/app.py --sweep`
  (occupancy × T grid). BLITZ root.
  See [`WETTER/docs/sparse_matrices.md`](../../WETTER/docs/sparse_matrices.md) § D.
- **external_posts/** – social/marketing snippets (e.g. LinkedIn)
- **tests/** – pytest unit and integration tests

Shared WETTER resources (outside BLITZ):

- [`converters/`](../../converters/) — format bridges → `.npy`
- [`EVT/`](../../EVT/) — Event reader (EVT3 archives → BLITZ via WOLKE; FUNKE is later)
- [`datasets/`](../../datasets/) — sample / reference datasets
