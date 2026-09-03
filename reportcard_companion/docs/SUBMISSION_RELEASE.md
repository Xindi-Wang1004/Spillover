# Genome Biology submission release checklist

Complete these steps **before** uploading to the Genome Biology portal.

## 1. Git tag (required)

```bash
cd /path/to/Spillover
git add transfer_GB/
git commit -m "GB estimand audit package v1.0 (stable submission)"
git tag -a gb-estimand-v1.0 -m "Genome Biology submission snapshot"
git push origin HEAD --tags
```

Record the tag commit SHA in `docs/REPRODUCIBILITY_LOCK.md` and regenerate hashes:

```bash
cd transfer_GB
python3 scripts/write_reproducibility_hashes.py
cp tables/Table_reproducibility_hashes.json ../00_to_upload/
```

## 2. Zenodo archive (required for Resource line)

1. Create a new Zenodo version from the tagged GitHub release (or upload `transfer_GB/` tarball).
2. Update manuscript Methods §4.7 with the new Zenodo DOI/version.
3. Keep checkpoint DOI `10.5281/zenodo.21809791` separate (model weights only).

## 3. PyPI (`genome-ml-reportcard`)

Wheel already built at `audit_toolkit/dist/`. After tag:

```bash
pip install twine
twine upload audit_toolkit/dist/*
```

See `audit_toolkit/docs/PYPI_RELEASE.md`. Manuscript cites `pip install genome-ml-reportcard`.

## 4. Verify before upload

- [ ] `Table_reproducibility_hashes.json` matches submission tag commit
- [ ] Cover letter date and corresponding emails
- [ ] Canonical manuscript = `GB_estimand_v1.md` (not `GB_manuscript_v1.md`)
- [ ] `00_to_upload/` docx rebuilt after final md edit
- [ ] Zenodo + GitHub links resolve publicly

## 5. Suggested cover positioning

Method / Resource — **estimand-first audit standard** for genome ML evaluation (not a viral discovery paper).
