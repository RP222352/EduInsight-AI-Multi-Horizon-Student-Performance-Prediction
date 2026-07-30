"""
spp_engine.py — all dashboard logic, no Streamlit (so it's testable/reusable).

THE IMPORTANT THING ABOUT THIS DATASET
--------------------------------------
"Academic Risk student performance Prediction.csv" is five different studies
stacked on top of each other (see the `source` column). Each study only fills
in its own columns; every other column holds a constant placeholder.

That breaks naive statistics badly. Example - tutoring:

    whole file        tutoring=0 -> 85% pass      tutoring=1 -> 52% pass
                      ("tutoring looks harmful")
    within StudentPerf, where the column actually varies:
                      tutoring=0 -> 44% pass      tutoring=1 -> 52% pass
                      (tutoring helps, +8 points)

The first version is Simpson's paradox: `tutoring=0` was pooling 9,131
placeholder rows from high-passing studies with 1,671 real ones. So every
benchmark, quantile, pass rate and counterfactual target in this engine is
computed inside the feature's own reference cohort.

Also produced here: confidence with an uncertainty interval, risk levels, SHAP
attributions, per-answer grading against the cohort, and recommendations
measured by re-running the model rather than looked up in a table.
Recommendations are split into immediate actions and long-term suggestions,
and an LLM can be plugged in later via Engine.set_advisor().
"""

from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder

from spp_features import FEATURE_META, KNOWN_LEAK_COLUMNS, meta

warnings.filterwarnings("ignore", category=FutureWarning)

FAIL_LABEL_INDEX = 0
TARGET_COL = "result"
SOURCE_COL = "source"

EXP_NAME = {"E1": "Late", "E2": "Early", "E3": "Very early"}
EXP_ORDER = {"E3": 0, "E2": 1, "E1": 2}
EXP_WINDOW = {
    "E3": "Start of term - enrolment and background data only.",
    "E2": "Mid-term - background plus engagement and behaviour.",
    "E1": "Late term - everything above plus actual grades.",
}

STATUS_ORDER = ["Excellent", "Good", "Moderate", "Needs improvement", "Critical",
                "Neutral", "Background", "Not recorded"]
STATUS_COLOR = {
    "Excellent": "#1e8449", "Good": "#58a55c", "Moderate": "#c9a227",
    "Needs improvement": "#d97706", "Critical": "#c0392b",
    "Neutral": "#7f8c8d", "Background": "#7f8c8d", "Not recorded": "#9aa5b1",
}
GOOD_STATUS = ("Excellent", "Good")
BAD_STATUS = ("Needs improvement", "Critical")

RISK_BANDS = [
    (0.20, "LOW", "#1e8449", "Not required",
     "Continue normal teaching; review at the next data point."),
    (0.40, "MODERATE", "#c9a227", "Monitor",
     "Keep under review; act now on any factor graded Critical."),
    (0.70, "HIGH", "#d97706", "Within one month",
     "Put a support plan in place this month and set a review date."),
    (1.01, "CRITICAL", "#c0392b", "Immediate",
     "Escalate this week: named staff member, agreed plan, contact home."),
]


def risk_band(fail_prob: float):
    for hi, name, colour, urgency, advice in RISK_BANDS:
        if fail_prob < hi:
            return dict(level=name, colour=colour, urgency=urgency, advice=advice)
    return dict(level="CRITICAL", colour="#c0392b", urgency="Immediate", advice="")


def label(f):         return meta(f)["label"]
def is_actionable(f): return bool(meta(f).get("actionable", True))
def describe(f):      return meta(f)["description"]
def immediate_of(f):  return list(meta(f).get("immediate", []) or [])
def longterm_of(f):   return list(meta(f).get("longterm", []) or [])


def pct_text(p: float, decimals: int = 0) -> str:
    """Never print 0% or 100%. A model that says a child has a 0% chance of
    passing is overstating what it can know, and nobody should read that
    sentence about a real student."""
    v = float(p) * 100
    if v >= 99.5:
        return ">99%"
    if v <= 0.5:
        return "<1%"
    return f"{v:.{decimals}f}%"


def _ord(n) -> str:
    n = int(round(n))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def load_any(p):
    try:
        import joblib
        return joblib.load(p)
    except Exception:
        with open(p, "rb") as f:
            return pickle.load(f)


# --------------------------------------------------------------------------- #
class Engine:
    #: optional callable(profile) -> {"immediate": [...], "longterm": [...]}
    _advisor = None

    @classmethod
    def set_advisor(cls, fn):
        """Plug in an LLM later. It receives the full student profile (values,
        statuses, SHAP drivers, measured counterfactuals) and returns action
        text. The numbers stay model-derived either way."""
        cls._advisor = fn

    def __init__(self, base_dir,
                 raw_csv_name="Academic Risk student performance Prediction.csv"):
        self.base_dir = base_dir
        path = os.path.join(base_dir, raw_csv_name)
        if not os.path.exists(path):
            for alt in (raw_csv_name.replace(" ", "_"), "Academic_Risk_student_performance_Prediction.csv"):
                p2 = os.path.join(base_dir, alt)
                if os.path.exists(p2):
                    path = p2
                    break
        self.raw = pd.read_csv(path)
        self.experiments = ["E1", "E2", "E3"]

        self.pass_mask, self.fail_mask, self.base_rate = self._outcome_masks()
        self.owners = self._detect_owners()

        self.models, self.features = {}, {}
        for exp in self.experiments:
            d = os.path.join(base_dir, "Results", exp)
            mp = os.path.join(d, "Best_Model.pkl")
            if not os.path.exists(mp):
                mp = os.path.join(d, "Models", "CatBoost.pkl")
            self.models[exp] = load_any(mp)
            self.features[exp] = list(load_any(os.path.join(d, "Feature_Names.pkl")))

        all_feats = sorted(set().union(*self.features.values()))
        self.specs = self._build_specs(all_feats)
        self.leaks = {f: s["leak_reason"] for f, s in self.specs.items() if s.get("leak")}

        e3, e2, e1 = self.features["E3"], self.features["E2"], self.features["E1"]
        self.wizard_order = (e3 + [f for f in e2 if f not in e3]
                                + [f for f in e1 if f not in e2])
        self.unlocks, seen = {}, set()
        for f in self.wizard_order:
            seen.add(f)
            for exp in ["E3", "E2", "E1"]:
                if exp not in self.unlocks.values() and set(self.features[exp]).issubset(seen):
                    self.unlocks[f] = exp

        self._explainers, self._sorted_cache = {}, {}

    # ------------------------------------------------------------------ setup
    def _outcome_masks(self):
        if TARGET_COL not in self.raw.columns:
            idx = self.raw.index
            return pd.Series(True, index=idx), pd.Series(False, index=idx), np.nan
        col = self.raw[TARGET_COL].astype(str).str.strip().str.lower()
        fail = col.isin(["fail", "failed", "0", "no", "dropout"])
        if not fail.any():
            fail = col == sorted(col.unique())[0]
        return ~fail, fail, float(fail.mean())

    def _detect_owners(self):
        """Which source study each column actually varies in. Everything else is
        placeholder padding and must be kept out of that column's statistics."""
        owners = {}
        if SOURCE_COL not in self.raw.columns:
            return owners
        n_src = self.raw[SOURCE_COL].nunique()
        for c in self.raw.columns:
            if c in (SOURCE_COL, TARGET_COL):
                continue
            try:
                nu = self.raw.groupby(SOURCE_COL)[c].nunique()
            except Exception:
                continue
            own = [s for s in nu.index if nu[s] > 1]
            if own and len(own) < n_src:
                owners[c] = own
        return owners

    def placeholder_value(self, feat):
        """The constant a merged CSV uses to pad a column in the studies that did
        not record it. A student carrying that value has not really been measured
        on this feature, so it must not be graded or acted on."""
        own = self.owners.get(feat)
        if not own or feat not in self.raw.columns:
            return None
        out = set()
        for src, grp in self.raw.loc[~self.raw[SOURCE_COL].isin(own)].groupby(SOURCE_COL):
            u = grp[feat].dropna().unique()
            if len(u) == 1:                       # this study padded the column
                out.add(u[0])
        return out or None

    def reference_mask(self, feat) -> pd.Series:
        own = self.owners.get(feat)
        if not own:
            return pd.Series(True, index=self.raw.index)
        return self.raw[SOURCE_COL].isin(own)

    def _build_specs(self, feats):
        specs = {}
        for f in feats:
            m = meta(f)
            ref = self.reference_mask(f)
            base = dict(label=m["label"], question=m["question"],
                        description=m["description"], unit=m.get("unit", ""),
                        actionable=bool(m.get("actionable", True)), prefer=m.get("prefer"),
                        immediate=list(m.get("immediate", []) or []),
                        longterm=list(m.get("longterm", []) or []),
                        in_data=f in self.raw.columns, owners=self.owners.get(f, []),
                        ref_n=int(ref.sum()), leak=False, leak_reason=None,
                        placeholder=self.placeholder_value(f))
            if f in KNOWN_LEAK_COLUMNS:
                base.update(leak=True, leak_reason=KNOWN_LEAK_COLUMNS[f])

            if f not in self.raw.columns:
                specs[f] = {**base, "kind": "numeric", "min": 0.0, "max": 100.0,
                            "default": 0.0, "step": 1.0, "is_int": True, "direction": 0,
                            "bands": [], "stats": {}, "auto_leak": None,
                            "range_text": "0 to 100 (column not present in the CSV; range is a guess)"}
                continue

            col = self.raw.loc[ref, f]
            levels = m.get("levels")
            numeric = pd.api.types.is_numeric_dtype(col)
            small_int = False
            if numeric and not levels:
                cc = pd.to_numeric(col, errors="coerce").dropna()
                small_int = bool(len(cc) and np.allclose(cc, cc.round())
                                 and cc.nunique() <= 12)
            if numeric and levels:
                specs[f] = {**base, **self._ordinal_spec(f, col, levels, ref)}
            elif small_int:
                # Quantile bands collapse to nonsense ("1 to 1", "below 0") when a
                # column only takes a handful of integer values, so show the real
                # pass rate at each value instead.
                cc = pd.to_numeric(col, errors="coerce").dropna()
                auto = {int(k): "" for k in sorted(cc.unique())}
                specs[f] = {**base, **self._ordinal_spec(f, col, auto, ref, bare=True)}
            elif numeric:
                specs[f] = {**base, **self._numeric_spec(f, col, ref)}
            else:
                specs[f] = {**base, **self._categorical_spec(f, ref)}

            if specs[f].get("auto_leak") and not specs[f]["leak"]:
                specs[f].update(leak=True, leak_reason=specs[f]["auto_leak"])
        return specs

    def _pf(self, ref):
        return self.pass_mask[ref], self.fail_mask[ref]

    @staticmethod
    def _auc(pos, neg):
        n1, n2 = len(pos), len(neg)
        if n1 < 5 or n2 < 5:
            return 0.5, 1.0
        allv = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
        r = pd.Series(allv).rank().to_numpy()
        auc = (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
        se = np.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2))
        return float(auc), float(se)

    def _numeric_spec(self, f, col, ref):
        s = pd.to_numeric(col, errors="coerce").dropna()
        is_int = bool(np.allclose(s, s.round()))
        pm, fm = self._pf(ref)
        ps = s[pm.reindex(s.index, fill_value=False)]
        fs = s[fm.reindex(s.index, fill_value=False)]
        enough = len(ps) >= 5 and len(fs) >= 5
        if not enough:
            ps = fs = s

        stats = dict(n=int(len(s)), mean=float(s.mean()), median=float(s.median()),
                     p10=float(s.quantile(.10)), p25=float(s.quantile(.25)),
                     p75=float(s.quantile(.75)), p90=float(s.quantile(.90)),
                     pass_median=float(ps.median()), pass_q25=float(ps.quantile(.25)),
                     pass_q75=float(ps.quantile(.75)), fail_median=float(fs.median()),
                     fail_q25=float(fs.quantile(.25)), fail_q75=float(fs.quantile(.75)),
                     pass_min=float(ps.min()), pass_max=float(ps.max()),
                     fail_min=float(fs.min()), fail_max=float(fs.max()))

        auc, se = self._auc(ps.to_numpy(), fs.to_numpy()) if enough else (0.5, 1.0)
        stats["auc"] = auc
        direction = 0 if abs(auc - .5) < max(.04, 2.5 * se) else (1 if auc > .5 else -1)

        auto_leak = None
        if enough and (stats["fail_max"] < stats["pass_min"] or stats["pass_max"] < stats["fail_min"]):
            auto_leak = (f"Failing students span {stats['fail_min']:g} to {stats['fail_max']:g} and "
                         f"passing students {stats['pass_min']:g} to {stats['pass_max']:g}, with no "
                         f"overlap at all. This column determines the outcome.")

        prefer = meta(f).get("prefer")
        contradicts = bool(prefer and direction and direction != prefer)
        if contradicts:
            direction = 0

        lo, hi = float(s.min()), float(s.max())
        step = 1.0 if is_int else round(max((hi - lo) / 100.0, 0.1), 2)
        unit = meta(f).get("unit", "")
        rt = (f"{lo:.0f} to {hi:.0f}" if is_int else f"{lo:.1f} to {hi:.1f}")
        if unit and unit not in ("category", "level", "yes / no", "band", "stage", "subject"):
            rt += f" {unit}"
        rt += f"  (cohort median {stats['median']:.4g})"

        return dict(kind="numeric", min=lo, max=hi, default=float(s.median()), step=step,
                    is_int=is_int, direction=direction, stats=stats, range_text=rt,
                    contradicts_prior=contradicts, auto_leak=auto_leak,
                    bands=self._numeric_bands(stats, direction, is_int))

    @staticmethod
    def _numeric_bands(stats, direction, is_int):
        if direction == 0:
            return []
        fmt = (lambda v: f"{v:.0f}") if is_int else (lambda v: f"{v:.1f}")
        pm, pq75, pq25 = stats["pass_median"], stats["pass_q75"], stats["pass_q25"]
        fm, fq25, fq75 = stats["fail_median"], stats["fail_q25"], stats["fail_q75"]
        if direction > 0:
            cuts = [("Excellent", f"{fmt(pq75)} and above", "top quartile of students who passed"),
                    ("Good", f"{fmt(pm)} to {fmt(pq75)}", "at or above the typical passing student"),
                    ("Moderate", f"{fmt(fm)} to {fmt(pm)}", "between the typical failing and passing student"),
                    ("Needs improvement", f"{fmt(fq25)} to {fmt(fm)}", "in the range where students usually fail"),
                    ("Critical", f"below {fmt(fq25)}", "worse than 75% of students who failed")]
        else:
            cuts = [("Excellent", f"{fmt(pq25)} and below", "better than most students who passed"),
                    ("Good", f"{fmt(pq25)} to {fmt(pm)}", "at or below the typical passing student"),
                    ("Moderate", f"{fmt(pm)} to {fmt(fm)}", "between the typical passing and failing student"),
                    ("Needs improvement", f"{fmt(fm)} to {fmt(fq75)}", "in the range where students usually fail"),
                    ("Critical", f"above {fmt(fq75)}", "worse than 75% of students who failed")]
        return [dict(status=s, range=r, meaning=mn) for s, r, mn in cuts]

    def _level_table(self, f, ref, keys, key_to_str):
        col = self.raw.loc[ref, f].astype(str)
        pm = self.pass_mask[ref]
        overall = float(pm.mean()) if len(pm) else np.nan
        rates, counts = {}, {}
        for k in keys:
            m = (col == key_to_str(k)).to_numpy()
            counts[k] = int(m.sum())
            rates[k] = float(pm.to_numpy()[m].mean()) if m.any() else np.nan
        valid = [k for k in keys if rates[k] == rates[k] and counts[k] > 0]
        separates = any(abs(rates[k] - overall) >
                        1.5 * np.sqrt(max(overall * (1 - overall), 1e-9) / max(counts[k], 1))
                        for k in valid) if valid else False
        auto_leak = None
        if len(valid) > 1 and all(rates[k] in (0.0, 1.0) for k in valid):
            auto_leak = ("Every category is either 100% pass or 100% fail, so this column "
                         "determines the outcome rather than predicting it.")
        return rates, counts, overall, separates, auto_leak

    def _bands_from_levels(self, keys, rates, counts, overall, separates, disp):
        bands = []
        for k in keys:
            r, n = rates.get(k, np.nan), counts.get(k, 0)
            if r != r or n == 0:
                bands.append(dict(status="Neutral", range=disp(k),
                                  meaning="no students at this level in the reference cohort"))
                continue
            lift = r - overall
            se = np.sqrt(max(overall * (1 - overall), 1e-9) / max(n, 1))
            if not separates or abs(lift) < 1.5 * se:
                st = "Neutral" if not separates else "Moderate"
            else:
                st = ("Excellent" if lift >= .10 else "Good" if lift >= .03 else
                      "Moderate" if lift > -.03 else
                      "Needs improvement" if lift > -.10 else "Critical")
            bands.append(dict(status=st, range=disp(k),
                              meaning=f"{r*100:.0f}% of these students pass "
                                      f"({lift*100:+.0f} pts vs the {overall*100:.0f}% cohort rate), n={n:,}"))
        return bands

    def _ordinal_spec(self, f, col, levels, ref, bare=False):
        """Coded 0-4 style fields: plain numbers to the model, labelled levels to a human."""
        s = pd.to_numeric(col, errors="coerce").dropna()
        keys = [k for k in sorted(levels) if (s == k).any()] or sorted(levels)
        rates, counts, overall, separates, auto_leak = self._level_table(
            f, ref, keys, lambda k: str(int(k)))
        pm, fm = self._pf(ref)
        auc, se = self._auc(s[pm.reindex(s.index, fill_value=False)].to_numpy(),
                            s[fm.reindex(s.index, fill_value=False)].to_numpy())
        direction = 0 if abs(auc - .5) < max(.04, 2.5 * se) else (1 if auc > .5 else -1)
        prefer = meta(f).get("prefer")
        contradicts = bool(prefer and direction and direction != prefer)
        if contradicts:
            direction = 0
        valid = [k for k in keys if rates[k] == rates[k]]
        disp = ((lambda k: str(int(k))) if bare
                else (lambda k: f"{int(k)} = {levels.get(k, k)}"))
        unit = meta(f).get("unit", "")
        rt = (f"{int(min(keys))} to {int(max(keys))}" + (f" {unit}" if unit and unit not in
              ("category", "level", "yes / no", "band") else "") if bare
              else "   ".join(f"{int(k)} = {levels[k]}" for k in sorted(levels)))
        return dict(kind="ordinal", bare=bare, min=float(min(keys)), max=float(max(keys)),
                    default=float(pd.Series(valid or keys).median()), step=1.0, is_int=True,
                    levels=levels, level_keys=keys, direction=direction,
                    contradicts_prior=contradicts, auto_leak=auto_leak,
                    stats=dict(pass_rate=rates, counts=counts, overall_pass=overall,
                               separates=separates, auc=auc, median=float(s.median()),
                               n=int(len(s))),
                    range_text=rt,
                    bands=self._bands_from_levels(keys, rates, counts, overall, separates, disp))

    def _categorical_spec(self, f, ref):
        col = self.raw[f].astype(str)
        cats = sorted(col.fillna("nan").unique().tolist())      # alphabetical == training
        le = LabelEncoder().fit(cats)
        present = sorted(self.raw.loc[ref, f].astype(str).unique().tolist())
        order = [c for c in (meta(f).get("order") or []) if c in present]
        display_order = order + [c for c in present if c not in order]
        rates, counts, overall, separates, auto_leak = self._level_table(
            f, ref, display_order, lambda k: str(k))
        return dict(kind="categorical", options=display_order, all_options=cats, encoder=le,
                    default=display_order[0] if display_order else cats[0], is_int=True,
                    direction=0, contradicts_prior=False, auto_leak=auto_leak,
                    stats=dict(pass_rate=rates, counts=counts, overall_pass=overall,
                               separates=separates),
                    range_text="One of: " + ", ".join(display_order),
                    bands=self._bands_from_levels(display_order, rates, counts, overall,
                                                  separates, lambda k: str(k)))

    # --------------------------------------------------------------- display
    def fmt(self, feat, raw_value) -> str:
        sp = self.specs[feat]
        if sp["kind"] == "categorical":
            return str(raw_value)
        if sp["kind"] == "ordinal":
            k = int(round(float(raw_value)))
            if sp.get("bare"):
                return str(k)
            return f"{k} = {sp['levels'].get(k, '?')}"
        v = float(raw_value)
        return f"{v:.0f}" if sp.get("is_int") else f"{v:.1f}"

    def question_help(self, feat) -> dict:
        sp = self.specs[feat]
        return dict(feature=feat, label=sp["label"], question=sp["question"],
                    description=sp["description"], unit=sp["unit"], kind=sp["kind"],
                    range_text=sp["range_text"], bands=sp["bands"],
                    actionable=sp["actionable"], direction=sp["direction"],
                    unlocks=self.unlocks.get(feat), leak=sp["leak"],
                    leak_reason=sp["leak_reason"], owners=sp["owners"], ref_n=sp["ref_n"],
                    contradicts=sp.get("contradicts_prior", False))

    # --------------------------------------------------------------- encoding
    def encode(self, raw_input: dict, issues: list = None) -> dict:
        note = issues.append if issues is not None else (lambda _x: None)
        out = {}
        src = raw_input.get(SOURCE_COL)
        if src is not None and str(src) != "nan":
            out["__source__"] = str(src)
        for f, spec in self.specs.items():
            if f == SOURCE_COL or f not in raw_input or raw_input[f] is None:
                continue
            v = raw_input[f]
            if isinstance(v, float) and np.isnan(v):
                continue
            if spec["kind"] == "categorical":
                sv = str(v)
                if sv not in list(spec["encoder"].classes_):
                    note(f"{f}: '{sv}' is not a category seen in training (expected one of "
                         f"{', '.join(map(str, spec['encoder'].classes_))}) - dropped")
                    continue
                out[f] = int(spec["encoder"].transform([sv])[0])
            else:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    note(f"{f}: '{v}' is not a number - dropped")
                    continue
                if spec.get("in_data") and not (spec["min"] <= fv <= spec["max"]):
                    note(f"{f}: {fv:g} is outside the observed range "
                         f"({spec['min']:g} to {spec['max']:g}) - the model is extrapolating")
                out[f] = fv
        return out

    def decode(self, feat, encoded_value):
        sp = self.specs[feat]
        if sp["kind"] == "categorical":
            try:
                return str(sp["encoder"].inverse_transform([int(round(encoded_value))])[0])
            except Exception:
                return str(encoded_value)
        return float(encoded_value)

    def runnable(self, encoded: dict):
        have = {k for k, v in encoded.items() if v is not None and k != "__source__"}
        return [e for e in self.experiments if set(self.features[e]).issubset(have)]

    def completeness(self, encoded, exp):
        fs = self.features[exp]
        return sum(1 for f in fs if f in encoded) / max(len(fs), 1)

    # ------------------------------------------------------------ prediction
    def _fail_index(self, model):
        classes = list(getattr(model, "classes_", [0, 1]))
        return classes.index(0) if 0 in classes else FAIL_LABEL_INDEX

    def _probs(self, rows, exp) -> np.ndarray:
        model, feats = self.models[exp], self.features[exp]
        X = pd.DataFrame([[r[f] for f in feats] for r in rows], columns=feats)
        return np.asarray(model.predict_proba(X))[:, self._fail_index(model)]

    def _explainer(self, exp):
        if exp not in self._explainers:
            self._explainers[exp] = shap.TreeExplainer(self.models[exp])
        return self._explainers[exp]

    def _sorted_col(self, feat):
        if feat not in self._sorted_cache:
            ref = self.reference_mask(feat)
            arr = pd.to_numeric(self.raw.loc[ref, feat], errors="coerce").dropna().to_numpy()
            self._sorted_cache[feat] = np.sort(arr)
        return self._sorted_cache[feat]

    # ------------------------------------------------------------ confidence
    def _stability(self, X, exp, fail_prob):
        model = self.models[exp]
        fi = self._fail_index(model)
        ps = []
        try:
            n = int(getattr(model, "tree_count_", 0))
            if n >= 30:
                # ntree_start does NOT give a prefix model - it drops the leading
                # trees. Sweep from 0 and keep the tail instead.
                period = max(1, n // 25)
                seq = [float(np.asarray(p)[0, fi]) for p in
                       model.staged_predict_proba(X, ntree_start=0, ntree_end=n,
                                                  eval_period=period)]
                ps = seq[-max(4, len(seq) // 3):]
        except Exception:
            ps = []
        if len(ps) < 3:
            try:
                ests = list(getattr(model, "estimators_", []))
                if len(ests) >= 10:
                    ps = [float(np.asarray(e.predict_proba(X))[0, fi]) for e in ests[:60]]
            except Exception:
                ps = []
        if len(ps) < 3:
            return 0.0, 0.0, 0
        side = fail_prob >= .5
        return float(np.std(ps)), float(np.mean([(p >= .5) != side for p in ps])), len(ps)

    def _typicality(self, encoded, exp):
        inside = checked = 0
        for f in self.features[exp]:
            sp = self.specs[f]
            if not sp.get("in_data") or f not in encoded:
                continue
            checked += 1
            if sp["kind"] in ("categorical", "ordinal"):
                inside += 1
            else:
                st = sp["stats"]
                lo = st["p10"] - 2.5 * (st["median"] - st["p10"] + 1e-9)
                hi = st["p90"] + 2.5 * (st["p90"] - st["median"] + 1e-9)
                inside += int(lo <= encoded[f] <= hi)
        return inside / checked if checked else 1.0

    def confidence(self, encoded, X, exp, fail_prob) -> dict:
        margin = abs(fail_prob - .5) * 2
        sd, flip, snaps = self._stability(X, exp, fail_prob)
        stability = float(np.clip(1 - 3 * flip - sd / .20, 0, 1)) if snaps else .75
        typical = self._typicality(encoded, exp)
        complete = self.completeness(encoded, exp)
        score = float(np.clip(.40 * margin + .30 * stability + .20 * typical + .10 * complete,
                              0, .97))
        level = "High" if score >= .72 else "Moderate" if score >= .50 else "Low"
        half = 1.96 * sd if snaps else .06
        lo, hi = float(np.clip(fail_prob - half, 0, 1)), float(np.clip(fail_prob + half, 0, 1))
        if margin < .15:
            note = ("This student sits almost exactly on the pass/fail boundary, so the verdict "
                    "could flip on a small change. Treat it as uncertain, not as a decision.")
        elif flip > 0:
            note = ("Some snapshots of the model disagreed with the final verdict - this student "
                    "is near the edge of what the model resolved cleanly.")
        elif typical < .7:
            note = ("Several answers fall outside the range the model was trained on, so it is "
                    "extrapolating rather than recognising a familiar pattern.")
        elif complete < .999:
            note = "Some inputs for this horizon are missing, which widens the uncertainty."
        elif stability < .6:
            note = ("The probability was still drifting as trees were added, though the verdict "
                    "itself never changed.")
        else:
            note = "Clear margin, stable across the ensemble, and typical of the training cohort."
        return dict(score=score, level=level, interval=(lo, hi), margin=float(margin),
                    stability=stability, flip_rate=float(flip), typicality=float(typical),
                    completeness=float(complete), note=note, boundary=bool(margin < .15))

    # ------------------------------------------------------------ benchmarks
    def assess(self, feat, encoded_value, source=None) -> dict:
        sp = self.specs[feat]
        raw_val = self.decode(feat, encoded_value)
        cohort_note = (f"compared against {sp['ref_n']:,} students from "
                       f"{', '.join(sp['owners'])}" if sp["owners"]
                       else f"compared against all {sp['ref_n']:,} students")
        out = dict(feature=feat, label=sp["label"], value=raw_val,
                   display=self.fmt(feat, raw_val), actionable=sp["actionable"],
                   leak=sp["leak"], cohort_note=cohort_note, ideal="-", percentile=None)

        if not sp.get("in_data"):
            return {**out, "status": "Neutral", "cohort": "-",
                    "comment": "not present in the reference dataset"}

        # If we know which study the row came from, we know for certain whether
        # this column was recorded - no need to guess from placeholder values.
        if source is not None and sp["owners"]:
            if str(source) not in [str(o) for o in sp["owners"]]:
                return {**out, "status": "Not recorded", "cohort": "-",
                        "comment": (f"the {source} study did not collect this field, so the "
                                    f"stored value is padding rather than a measurement "
                                    f"(only {', '.join(sp['owners'])} recorded it)")}
            ph = None
        else:
            ph = sp.get("placeholder")
        if ph and sp["owners"]:
            def _same(a, b):
                try:
                    return abs(float(a) - float(b)) < 1e-9
                except (TypeError, ValueError):
                    return str(a) == str(b)
            if any(_same(raw_val, x) for x in ph):
                return {**out, "status": "Not recorded", "cohort": "-",
                        "comment": (f"this value matches the placeholder used when a study did "
                                    f"not record the field, so it is probably not a real "
                                    f"measurement (only {', '.join(sp['owners'])} collected it)")}

        if sp["kind"] in ("categorical", "ordinal"):
            rates, overall = sp["stats"]["pass_rate"], sp["stats"]["overall_pass"]
            key = int(round(float(raw_val))) if sp["kind"] == "ordinal" else str(raw_val)
            valid = {k: v for k, v in rates.items() if v == v}
            best = max(valid, key=valid.get) if valid else None
            ideal = self.fmt(feat, best) if best is not None else "-"
            if not sp["stats"].get("separates", True):
                return {**out, "status": "Neutral", "ideal": ideal,
                        "comment": "pass rates do not differ meaningfully between levels here",
                        "cohort": f"cohort pass rate {overall*100:.0f}%"}
            r = rates.get(key, np.nan)
            if r != r:
                return {**out, "status": "Neutral", "ideal": ideal, "cohort": "-",
                        "comment": "no students at this level in the reference cohort"}
            lift, n = r - overall, sp["stats"]["counts"].get(key, 0)
            se = np.sqrt(max(overall * (1 - overall), 1e-9) / max(n, 1))
            st = ("Moderate" if abs(lift) < 1.5 * se else
                  "Excellent" if lift >= .10 else "Good" if lift >= .03 else
                  "Moderate" if lift > -.03 else
                  "Needs improvement" if lift > -.10 else "Critical")
            return {**out, "status": st, "ideal": ideal,
                    "comment": (f"{r*100:.0f}% of students at this level pass, against a "
                                f"{overall*100:.0f}% cohort rate ({lift*100:+.0f} pts, n={n:,})"),
                    "cohort": f"best level: {ideal}"}

        stt, v = sp["stats"], float(raw_val)
        arr = self._sorted_col(feat)
        pct = float(np.searchsorted(arr, v, side="right") / max(len(arr), 1) * 100)
        d = sp["direction"]
        if d == 0:
            why = ("the data points the opposite way to common sense here, so it is left ungraded"
                   if sp.get("contradicts_prior") else
                   "this field does not separate passing from failing students in its cohort")
            return {**out, "status": "Neutral", "percentile": pct, "comment": why,
                    "cohort": f"median {stt['median']:.4g}"}
        if d > 0:
            st = ("Excellent" if v >= stt["pass_q75"] else "Good" if v >= stt["pass_median"] else
                  "Moderate" if v >= stt["fail_median"] else
                  "Needs improvement" if v >= stt["fail_q25"] else "Critical")
            ideal = f"{stt['pass_q75']:.4g} or above"
        else:
            st = ("Excellent" if v <= stt["pass_q25"] else "Good" if v <= stt["pass_median"] else
                  "Moderate" if v <= stt["fail_median"] else
                  "Needs improvement" if v <= stt["fail_q75"] else "Critical")
            ideal = f"{stt['pass_q25']:.4g} or below"
        return {**out, "status": st, "percentile": pct, "ideal": ideal,
                "comment": (f"{_ord(pct)} percentile of its cohort; students who pass typically "
                            f"reach {stt['pass_median']:.4g}"),
                "cohort": f"pass median {stt['pass_median']:.4g}  fail median {stt['fail_median']:.4g}"}

    # -------------------------------------------------- counterfactual engine
    def _targets_for(self, feat, current, source=None):
        sp = self.specs[feat]
        if not sp["actionable"] or not sp.get("in_data") or sp["leak"]:
            return []
        if source is not None and sp["owners"] and \
                str(source) not in [str(o) for o in sp["owners"]]:
            return []                  # never act on a field this study never recorded
        # If we know which study the row came from, we know for certain whether
        # this column was recorded - no need to guess from placeholder values.
        if source is not None and sp["owners"]:
            if str(source) not in [str(o) for o in sp["owners"]]:
                return {**out, "status": "Not recorded", "cohort": "-",
                        "comment": (f"the {source} study did not collect this field, so the "
                                    f"stored value is padding rather than a measurement "
                                    f"(only {', '.join(sp['owners'])} recorded it)")}
            ph = None
        else:
            ph = sp.get("placeholder")
        if ph and sp["owners"]:
            cur_raw = self.decode(feat, current)
            for x in ph:
                try:
                    match = abs(float(cur_raw) - float(x)) < 1e-9
                except (TypeError, ValueError):
                    match = str(cur_raw) == str(x)
                if match:
                    return []      # never "improve" a value that was never measured
        if sp["kind"] in ("categorical", "ordinal"):
            if not sp["stats"].get("separates", True):
                return []
            rates = sp["stats"]["pass_rate"]
            cur_key = (int(round(float(current))) if sp["kind"] == "ordinal"
                       else str(self.decode(feat, current)))
            cur_r = rates.get(cur_key, np.nan)
            out = []
            for k, r in sorted(rates.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1)):
                if k == cur_key or r != r:
                    continue
                if cur_r == cur_r and r <= cur_r + .01:
                    continue
                enc = (float(k) if sp["kind"] == "ordinal"
                       else int(sp["encoder"].transform([str(k)])[0]))
                out.append((enc, f"change from {self.fmt(feat, cur_key)} to {self.fmt(feat, k)}",
                            "switch"))
                if len(out) >= 2:
                    break
            return out

        d = sp["direction"]
        if d == 0:
            return []
        stt, cur, step = sp["stats"], float(current), sp["step"]
        rnd = (lambda x: float(round(x))) if sp["is_int"] else (lambda x: float(round(x / step) * step))
        mid = stt["pass_median"]
        goal = stt["pass_q75"] if d > 0 else stt["pass_q25"]
        cands = []
        for val in (cur + .5 * (mid - cur), mid, goal):
            val = rnd(np.clip(val, sp["min"], sp["max"]))
            if (d > 0 and val <= cur + step * .5) or (d < 0 and val >= cur - step * .5):
                continue
            if any(abs(val - v) < step * .5 for v, _, _ in cands):
                continue
            unit = (f" {sp['unit']}" if sp["unit"] and
                    sp["unit"] not in ("category", "level", "band", "yes / no") else "")
            cands.append((val, f"move from {self.fmt(feat, cur)} to {self.fmt(feat, val)}{unit}",
                          "target"))
        return cands[:2]

    def _candidate_rows(self, base_row, current_row, feats, skip, source=None):
        rows, metas = [], []
        for f in feats:
            if f in skip:
                continue
            for val, text, kind in self._targets_for(f, base_row[f], source=source):
                r = dict(current_row)
                r[f] = val
                rows.append(r)
                metas.append(dict(feature=f, label=self.specs[f]["label"], to_encoded=val,
                                  to_display=self.fmt(f, self.decode(f, val)),
                                  from_display=self.fmt(f, self.decode(f, base_row[f])),
                                  change_text=text, kind=kind))
        return rows, metas

    def what_if(self, encoded, exp, top_n=6, max_steps=5, source=None) -> dict:
        source = source if source is not None else encoded.get("__source__")
        feats = self.features[exp]
        base = {f: encoded[f] for f in feats}
        base_p = float(self._probs([base], exp)[0])
        rows, metas = self._candidate_rows(base, base, feats, skip=set(), source=source)
        if not rows:
            return dict(base_prob=base_p, levers=[], plan=None, steps=[])

        lg = lambda p: float(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))))
        base_l = lg(base_p)
        probs = self._probs(rows, exp)
        by_f = {}
        for m, p in zip(metas, probs):
            lv = dict(m, new_prob=float(p), drop=base_p - float(p),
                      logit_drop=base_l - lg(float(p)))
            if lv["feature"] not in by_f or lv["drop"] > by_f[lv["feature"]]["drop"]:
                by_f[lv["feature"]] = lv
        levers = sorted((lv for lv in by_f.values() if lv["drop"] > .002 or lv["logit_drop"] > .10),
                        key=lambda x: (-x["drop"], -x["logit_drop"]))[:top_n]

        cur, cur_p, used, steps = dict(base), base_p, set(), []
        for _ in range(max_steps):
            rws, mts = self._candidate_rows(base, cur, feats, skip=used, source=source)
            if not rws:
                break
            ps = self._probs(rws, exp)
            j = int(np.argmin(ps))
            new_p, gain = float(ps[j]), cur_p - float(ps[j])
            if gain <= .002 and (lg(cur_p) - lg(new_p)) <= .10:
                break
            m = mts[j]
            steps.append(dict(m, step=len(steps) + 1, from_prob=cur_p, new_prob=new_p, drop=gain))
            cur = dict(cur)
            cur[m["feature"]] = m["to_encoded"]
            cur_p = new_p
            used.add(m["feature"])
        if steps and (base_p - cur_p) > .02:
            while len(steps) > 1 and steps[-1]["drop"] < .002:
                steps.pop()
                cur_p = steps[-1]["new_prob"]

        plan = (dict(base_prob=base_p, new_prob=cur_p, drop=base_p - cur_p,
                     actions=[s["feature"] for s in steps], steps=steps) if steps else None)
        return dict(base_prob=base_p, levers=levers, plan=plan, steps=steps)

    # ---------------------------------------------------------------- explain
    def explain(self, encoded: dict, exp: str, source=None) -> dict:
        source = source if source is not None else encoded.get("__source__")
        model, feats = self.models[exp], self.features[exp]
        X = pd.DataFrame([{f: encoded[f] for f in feats}], columns=feats)
        fi = self._fail_index(model)
        fail_prob = float(np.asarray(model.predict_proba(X))[0, fi])

        sv = self._explainer(exp)(X)
        vals, base = np.asarray(sv.values), np.asarray(sv.base_values)
        if vals.ndim == 3:
            fv, fb = vals[0, :, fi], float(base[0, fi])
        else:
            fv, fb = (vals[0], float(base[0])) if fi == 1 else (-vals[0], -float(base[0]))
        fv = np.asarray(fv, dtype=float)

        dfv = pd.DataFrame({"feature": feats, "shap": fv, "value": [encoded[f] for f in feats]})
        dfv = dfv.reindex(dfv.shap.abs().sort_values(ascending=False).index)

        factors = []
        for _, row in dfv.iterrows():
            a = self.assess(row.feature, row.value, source=source)
            # Background fields (age, gender, income...) still carry model weight,
            # but grading them Excellent/Critical implies an action nobody can take.
            if not a["actionable"]:
                a["graded_status"] = a["status"]
                a["status"] = "Background"
            a["shap"] = float(row.shap)
            a["effect"] = ("raises risk" if row.shap > .001 else
                           "lowers risk" if row.shap < -.001 else "no effect")
            a["influence"] = ("High" if abs(row.shap) >= .5 else
                              "Medium" if abs(row.shap) >= .15 else "Low")
            factors.append(a)

        rank = {s: i for i, s in enumerate(STATUS_ORDER)}
        factors = [f for f in factors]
        strengths = sorted([f for f in factors if f["status"] in GOOD_STATUS],
                           key=lambda f: (rank[f["status"]], f["shap"]))
        concerns = sorted([f for f in factors if f["status"] in BAD_STATUS],
                          key=lambda f: (-rank[f["status"]], -f["shap"]))
        watch = sorted([f for f in factors if f["status"] == "Moderate"], key=lambda f: -f["shap"])
        critical = [f for f in factors if f["status"] == "Critical"]

        conf = self.confidence(encoded, X, exp, fail_prob)
        wi = self.what_if(encoded, exp, source=source)
        recs = self._recommendations(wi, factors)
        immediate, longterm = self._actions(recs, concerns, factors, fail_prob)

        expl = shap.Explanation(values=fv, base_values=fb, data=X.iloc[0].values,
                                feature_names=[label(f) for f in feats])
        band = risk_band(fail_prob)
        return {
            "exp": exp, "exp_name": EXP_NAME.get(exp, exp), "window": EXP_WINDOW.get(exp, ""),
            "fail_prob": fail_prob, "pass_prob": 1 - fail_prob,
            "verdict": "AT RISK" if fail_prob >= .5 else "On track",
            "risk_level": band["level"], "risk_colour": band["colour"],
            "urgency": band["urgency"], "urgency_advice": band["advice"],
            "confidence": conf,
            "risk": [(f["label"], f["shap"]) for f in factors if f["shap"] > 0][:5],
            "protective": [(f["label"], f["shap"]) for f in factors if f["shap"] < 0][:5],
            "factors": factors, "strengths": strengths, "concerns": concerns,
            "watch": watch, "critical": critical,
            "what_if": wi, "recommendations": recs,
            "immediate": immediate, "suggestions": longterm,
            "background": [f["label"] for f in factors if not f["actionable"] and f["shap"] > 0][:5],
            "leaks_used": [f["label"] for f in factors if f["leak"]],
            "explanation": expl,
        }

    @staticmethod
    def _recommendations(wi, factors):
        status_of = {f["feature"]: f["status"] for f in factors}
        shap_of = {f["feature"]: f["shap"] for f in factors}
        alone = {lv["feature"]: lv for lv in wi.get("levers", [])}
        source = wi.get("steps") or list(alone.values())
        out = []
        for i, s in enumerate(source, 1):
            f = s["feature"]
            solo = alone.get(f)
            frm = s.get("from_prob", wi["base_prob"])
            out.append(dict(
                rank=i, feature=f, label=s["label"],
                action=f"{s['label']}: {s['change_text']}",
                current=s["from_display"], target=s["to_display"],
                from_prob=frm, to_prob=s["new_prob"], drop=s["drop"],
                impact_text=(f"failure risk {frm*100:.0f}% to {s['new_prob']*100:.0f}% "
                             f"({-s['drop']*100:+.0f} pts)" if s["drop"] >= .005 else
                             f"no visible change alone at {frm*100:.0f}% risk - it is the "
                             f"combination that moves the number"),
                solo_drop=(solo["drop"] if solo else None),
                solo_text=(("%.0f pts on its own" % (solo["drop"] * 100))
                           if solo and solo["drop"] >= .01 else
                           ("helps only in combination" if solo else "-")),
                status=status_of.get(f, "-"), shap=shap_of.get(f, 0.0),
                immediate=immediate_of(f), longterm=longterm_of(f)))
        return out

    def _actions(self, recs, concerns, factors, fail_prob):
        """Immediate actions vs long-term suggestions. Replaced wholesale if an
        LLM advisor is registered via Engine.set_advisor()."""
        if Engine._advisor is not None:
            try:
                got = Engine._advisor(dict(fail_prob=fail_prob, factors=factors,
                                           recommendations=recs, concerns=concerns))
                if got:
                    return (list(got.get("immediate", []))[:6],
                            list(got.get("longterm", []))[:6])
            except Exception:
                pass
        imm, lng = [], []
        for r in recs[:4]:
            for a in r["immediate"]:
                if a not in imm:
                    imm.append(a)
            for a in r["longterm"]:
                if a not in lng:
                    lng.append(a)
        for f in concerns[:4]:
            for a in immediate_of(f["feature"]):
                if a not in imm:
                    imm.append(a)
            for a in longterm_of(f["feature"]):
                if a not in lng:
                    lng.append(a)
        return imm[:6], lng[:6]

    def explain_all(self, encoded: dict, source=None):
        source = source if source is not None else encoded.get("__source__")
        return {e: self.explain(encoded, e, source=source) for e in self.runnable(encoded)}

    @staticmethod
    def overall(results: dict) -> dict:
        if not results:
            return {}
        order = [e for e in ["E1", "E2", "E3"] if e in results]
        head = results[order[0]]
        probs = [results[e]["fail_prob"] for e in order]
        spread = float(max(probs) - min(probs)) if len(probs) > 1 else 0.0
        agree = len({results[e]["verdict"] for e in order}) == 1
        conf = head["confidence"]["score"]
        if len(order) > 1:
            conf = float(np.clip(conf + (.05 if agree else -.08) - .15 * spread, 0, .97))
        band = risk_band(head["fail_prob"])
        return dict(exp=head["exp"], exp_name=head["exp_name"], fail_prob=head["fail_prob"],
                    pass_prob=head["pass_prob"], verdict=head["verdict"],
                    risk_level=band["level"], risk_colour=band["colour"],
                    urgency=band["urgency"], urgency_advice=band["advice"],
                    interval=head["confidence"]["interval"], confidence=conf,
                    level="High" if conf >= .72 else "Moderate" if conf >= .50 else "Low",
                    agree=agree, spread=spread, horizons=order, note=head["confidence"]["note"])

    # ---------------------------------------------------------- plain summaries
    @staticmethod
    def narrative(res):
        c = res["confidence"]
        lo, hi = c["interval"]
        tail = (f"Confidence: {c['level']} ({c['score']*100:.0f}/100); likely range "
                f"{pct_text(lo)}-{pct_text(hi)}.")
        if res["verdict"] == "AT RISK":
            drv = "; ".join(n for n, _ in res["risk"][:3])
            s = (f"Predicted AT RISK of failing - {pct_text(res['fail_prob'])} failure "
                 f"probability ({pct_text(res['pass_prob'])} chance of passing). Risk level "
                 f"{res['risk_level']}. Main drivers: {drv}. {tail}")
            if res["recommendations"]:
                s += f" Highest-impact change: {res['recommendations'][0]['action']}."
        else:
            prt = "; ".join(n for n, _ in res["protective"][:3])
            s = (f"Predicted on track to pass - {pct_text(res['pass_prob'])} pass likelihood. "
                 f"Risk level {res['risk_level']}. Supported mainly by: {prt}. {tail}")
            if res["concerns"]:
                s += " Still worth watching: " + ", ".join(f["label"] for f in res["concerns"][:2]) + "."
        return s

    @staticmethod
    def audience_summaries(res, student_id="This student"):
        """Same facts, three readerships. Deliberately avoids stating a 0% or 100%
        chance about a real child, and gives each reader actions they can take."""
        fp, pp = res["fail_prob"], res["pass_prob"]
        good = [f["label"] for f in res["strengths"][:3]]
        bad = [f["label"] for f in res["concerns"][:3]]
        recs = res.get("recommendations") or []
        gtxt = ", ".join(good) if good else "no factor currently above the passing benchmark"
        btxt = ", ".join(bad) if bad else "no factor currently in the failing range"
        plan = res["what_if"].get("plan")

        teacher = (f"{student_id} carries a {pct_text(fp)} modelled probability of failing "
                   f"(risk level {res['risk_level']}, confidence {res['confidence']['level']}). "
                   f"Strongest areas: {gtxt}. Weakest: {btxt}. Intervention priority is "
                   f"{res['urgency'].lower()}. {res['urgency_advice']}")
        if plan:
            teacher += (f" The modelled action plan would move the failure probability from "
                        f"{pct_text(plan['base_prob'])} to {pct_text(plan['new_prob'])}.")

        # parent: no "X in 100" at the extremes, and a concrete first step
        if fp >= .995 or pp >= .995:
            odds = ("On the information provided, the model is very confident about the "
                    "current direction of travel - but a prediction is not a result, and it "
                    "is based only on the data entered here.")
        else:
            odds = (f"On the information provided, there is roughly a {fp*100:.0f} in 100 "
                    f"chance of not passing this term and a {pp*100:.0f} in 100 chance of "
                    f"passing.")
        parent = (f"{odds} What is going well: {gtxt}. What needs attention: {btxt}. ")
        if recs:
            r0 = recs[0]
            parent += (f"The single change the model responds to most is {r0['label'].lower()}: "
                       f"moving from {r0['current']} to {r0['target']}. Please talk to the "
                       f"school about how to support that.")
        else:
            parent += "Please speak to the school about the right next step."

        # student: never quote a 0% chance; lead with what is in their control
        if pp <= .05:
            opener = ("Right now the pattern in your data looks like students who struggled - "
                      "but this is a prediction from numbers, not a decision about you, and "
                      "the numbers move when the work changes.")
        else:
            opener = (f"Right now the model puts your chance of passing at about "
                      f"{pct_text(pp)}. That is a prediction from data, not a decision about "
                      f"you, and it can change.")
        student = f"{opener} You are doing well on: {gtxt}. What is holding you back: {btxt}. "
        if recs:
            r0 = recs[0]
            student += (f"The thing that would shift this most is {r0['label'].lower()}: "
                        f"from {r0['current']} to {r0['target']}.")
            if plan and plan["drop"] > .05:
                student += (f" If you and the school work through the full plan, the model's "
                            f"estimate moves to {pct_text(1 - plan['new_prob'])} chance of "
                            f"passing.")
        else:
            student += "Ask your teacher to go through this report with you and agree a plan."
        return dict(teacher=teacher, parent=parent, student=student)
