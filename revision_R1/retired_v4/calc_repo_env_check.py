"""Provenance checks for the reporting items that are not analysis numbers (follow-up review):
(1) the public code repository cited in the submission, read at its current head through the GitHub API (read-only):
    commits, whether the KNHANES script applies one uniform map to every semi-quantitative field (M2), whether it filters
    age >= 19, the leukocyte esterase constant, and whether the training script parses;
(2) the software environment of the locked models: library versions recorded in the saved model files and whether they
    load without version-mismatch warnings in the revision environment, plus the revision environment's versions.
Writes fu_repository_check.csv and fu_environment.csv. Run: /opt/anaconda3/bin/python calc_repo_env_check.py
(requires the GitHub CLI 'gh'; the repository section is skipped if it is unavailable)."""
import os, sys, json, ast, base64, re, shutil, subprocess, platform, warnings
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "eunchanj/Urine-Dipstick-Hyperuricemia-Screening"
rows = []


def gh(path):
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


if shutil.which("gh"):
    commits = gh(f"repos/{REPO}/commits")
    rows.append(dict(item="commits on the default branch", value=len(commits)))
    for c in commits:
        rows.append(dict(item="commit", value=f"{c['sha'][:7]} {c['commit']['author']['date'][:10]} {c['commit']['message'].splitlines()[0]}"))
    rows.append(dict(item="tags", value=len(gh(f"repos/{REPO}/tags"))))
    rows.append(dict(item="releases", value=len(gh(f"repos/{REPO}/releases"))))

    def src(path):
        return base64.b64decode(gh(f"repos/{REPO}/contents/{path}")["content"]).decode("utf-8")

    kp = src("data/knhanes_processing.py")
    m = re.search(r"KNHANES_TO_MODEL\s*=\s*(\{[^}]*\})", kp)
    rows.append(dict(item="knhanes_processing.py: map applied to glucose, protein, ketone, bilirubin, occult blood, urobilinogen",
                     value=m.group(1) if m else "not found"))
    fields = re.search(r"DIPSTICK_KN_TO_MODEL\s*=\s*\{([^}]*)\}", kp)
    rows.append(dict(item="knhanes_processing.py: fields mapped with that single map",
                     value="; ".join(re.findall(r'"(HE_\w+)"', fields.group(1))) if fields else "not found"))
    rows.append(dict(item="knhanes_processing.py: age >= 19 filter present", value=bool(re.search(r'\[\s*"age"\s*\]\s*>=\s*19', kp))))
    le = re.search(r"le_mean:\s*float\s*=\s*([\d.]+)", kp)
    rows.append(dict(item="knhanes_processing.py: leukocyte esterase constant", value=le.group(1) if le else "not found"))
    tr = src("src/train_uric.py")
    try:
        ast.parse(tr); ok = "parses"
    except SyntaxError as e:
        ok = f"SyntaxError: {e.msg} at line {e.lineno}"
    rows.append(dict(item="src/train_uric.py: Python parse", value=ok))
    rows.append(dict(item="src/train_uric.py: outcome column name used", value=";".join(sorted(set(re.findall(r'"col":\s*"(\w+)"', tr))))))
    du = src("src/data_utils.py")
    rows.append(dict(item="src/data_utils.py: BMI feature name", value="BMI" if "'BMI'" in du and "bmi_unified" not in du else "other"))
    req = src("requirements.txt")
    for pkg in ("scikit-learn", "lightgbm", "xgboost", "catboost", "shap"):
        mm = re.search(rf"(?m)^{pkg}==([\w.]+)", req)
        rows.append(dict(item=f"requirements.txt: {pkg}", value=mm.group(1) if mm else "not pinned"))
    py = re.search(r"Python\s*([\d.]+)", req)
    rows.append(dict(item="requirements.txt: Python", value=py.group(1) if py else "not stated"))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "fu_repository_check.csv"), index=False)
    for r in rows:
        print(r["item"], "|", r["value"])

# ---------------------------------------------------------------- environment
sys.path.insert(0, HERE)
import cohort  # noqa: E402
import xgboost, lightgbm, catboost, sklearn, shap, numpy, scipy, statsmodels  # noqa: E402
env = [dict(item="revision environment: Python", value=platform.python_version())]
for mod in (sklearn, lightgbm, xgboost, catboost, shap, numpy, pd, scipy, statsmodels):
    env.append(dict(item=f"revision environment: {mod.__name__}", value=mod.__version__))
for t in cohort.TARGETS:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = cohort.load_model(t)
    inc = [x for x in w if "Inconsistent" in type(x.message).__name__ or "InconsistentVersion" in str(x.category)]
    env.append(dict(item=f"{t}: scikit-learn version-mismatch warnings on loading", value=len(inc)))
    xg = m.calibrated_classifiers_[0].estimator.estimators_[1]
    ver = json.loads(xg.get_booster().save_config()).get("version")
    env.append(dict(item=f"{t}: XGBoost version recorded in the saved booster", value=".".join(map(str, ver))))
    lg = m.calibrated_classifiers_[0].estimator.estimators_[0]
    env.append(dict(item=f"{t}: LightGBM model format version", value=lg.booster_.dump_model().get("version")))
pd.DataFrame(env).to_csv(os.path.join(HERE, "fu_environment.csv"), index=False)
for r in env:
    print(r["item"], "|", r["value"])
