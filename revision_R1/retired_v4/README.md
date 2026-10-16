# Retired scripts (original models, bestmodel_v4)

These three scripts were used in the first part of the revision, when the submitted (original) models were
re-evaluated as saved. They were retired on 2026-10-17, after the models were retrained (bestmodel_v5): the revised
manuscript and Multimedia Appendix 1 report only the retrained models, and none of these scripts' outputs appears in
them. The scripts are kept for the audit history that the response letter describes (the reproduction of the
submitted results, the nitrite parsing error and the state of this repository at commit 2d8b5b0). They are the
authors' last versions, with the same path changes as the other scripts (environment variables with the original
value as the fallback; see the top-level README).

| Script | What it did | Why it was retired |
|---|---|---|
| `calc_original.py` | Reproduced the submitted internal and external AUROC and Brier score of the original models exactly from the saved model files and `performance.csv` of `src/train_uric.py` (`PROJ_U_RESULTS`), then evaluated them on the person-clean adult internal test rows and on KNHANES adults. | The revised article reports the retrained models only. The comparison of the original and retrained models on the same persons, for the response letter, is made by `../calc_main.py` (`original_vs_retrained.csv`). |
| `calc_nitrite.py` | Recovered the nitrite results recorded as "Positive"/"POSITIVE" in the raw first-visit file, which the processed file had read as missing, and evaluated the original models on the recovered persons. | The corrected cohort reads nitrite from the raw text (`cohort.recover_nitrite`), and `../build_cohort_v2.py` checks it against the validated data cache. |
| `calc_repo_env_check.py` | Read this repository's default branch (then at commit 2d8b5b0) through the GitHub API (read-only) and the software versions recorded in the original model files. | It concerned the original models and the earlier repository. The environment of the retrained models is recorded in `models/bestmodel_v5/config.json` (see `requirements.txt`). |

They are not part of the run order (`../README.md`). Do not run `calc_original.py`: it writes
`preds_internal_clean.csv` and `preds_external_corrected.csv`, the prediction files that `calc_main.py` writes for
the retrained models. The scripts import `cohort.py` and `common.py` from the parent folder; they were last run on
2026-10-12 (`calc_original.py`, `calc_nitrite.py`) and 2026-09-30 (`calc_repo_env_check.py`), with the `cohort.py` of
that time. With the current `cohort.py`, `load_model()` returns the retrained models unless `URIC_MODEL_SET=v4` is set.
