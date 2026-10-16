# Frozen training record of the retrained models

`config.json` was written by `train_v5.py` on 2026-10-16 at 23:09:07, before the retrained models were fitted and
before any of their internal test or KNHANES results was computed. It records the inputs, the cohort steps, the
split, the training recipe and search spaces, the library versions, and SHA-256 checksums of the training inputs,
of the training and test person lists, and of the three code files that defined the data and the training.

The three code files are stored here exactly as they were at the freeze (byte for byte; they keep the authors'
absolute paths), so that the checksums can be verified:

```bash
shasum -a 256 cohort.py train_v5.py run_uric_module_v4.py
# compare with config.json -> sha256 -> code
```

The working versions in `revision_R1/` differ only in that data, model and output locations are read from
environment variables (see `revision_R1/README.md`) and, for `cohort.py`, in one function added after the freeze
(`persons_v2`, used by the missing-data and verification-bias analyses) and in its module docstring.
`run_uric_module_v4.py` is the training script of the submitted models (published here as `src/train_uric.py`
with path variables); `train_v5.py` copies its `tune_and_build` recipe line for line.
