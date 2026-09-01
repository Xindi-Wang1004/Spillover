# PyPI release (`genome-ml-reportcard`)

## Local verify

```bash
cd audit_toolkit
pip install -e ".[dev]"
python tests/test_smoke.py
pip install build
python -m build
```

## Publish (maintainer)

1. Bump `[project].version` in `pyproject.toml`.
2. Create GitHub release tag `reportcard-vX.Y.Z`.
3. Upload with PyPI token:

```bash
pip install twine
twine upload dist/*
```

4. Users install with:

```bash
pip install genome-ml-reportcard
genome-ml-reportcard --help
```

Zenodo: archive the same tag alongside the full `transfer_GB/` paper bundle.
