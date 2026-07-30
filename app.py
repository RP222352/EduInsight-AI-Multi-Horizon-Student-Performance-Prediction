"""
Student Performance Prediction — dashboard
Run:  streamlit run app.py
Place this file at the TOP LEVEL of your project folder (next to Results/ and the raw CSV).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from ai_pdf_report import AIReportPDF
from report_formatter import ReportFormatter
from llm_report import LLMReportGenerator
from pdf_report import build_pdf
from spp_engine import (EXP_NAME, EXP_WINDOW, STATUS_COLOR, Engine, label,
                        pct_text)

ai_pdf_generator = AIReportPDF()
st.set_page_config(page_title="Student Risk Predictor", page_icon="🎓", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource(show_spinner="Loading models…")
def get_engine():
    return Engine(BASE_DIR)


@st.cache_resource(show_spinner=False)
def get_llm():
    return LLMReportGenerator(model="llama3:latest")


try:
    E = get_engine()
except Exception as e:
    st.error(f"Could not load models/raw CSV from this folder.\n\n{e}")
    st.stop()

llm = get_llm()


def wide(fn, *a, **kw):
    try:
        return fn(*a, width="stretch", **kw)
    except TypeError:
        return fn(*a, use_container_width=True, **kw)


def chip(status):
    c = STATUS_COLOR.get(status, "#6b7280")
    return (f"<span style='background:{c};color:#fff;padding:2px 8px;border-radius:10px;"
            f"font-size:0.75rem;white-space:nowrap'>{status}</span>")


def risk_chip(level, colour):
    return (f"<span style='background:{colour};color:#fff;padding:3px 12px;border-radius:12px;"
            f"font-weight:700;font-size:0.85rem'>{level}</span>")


# ============================================================ question helper
def question_panel(feat):
    q = E.question_help(feat)
    st.markdown(f"### {q['label']}")
    tags = [f"`{feat}`"]
    if not q["actionable"]:
        tags.append("background factor — context only, not a target for action")
    if q["leak"]:
        tags.append("⚠️ target leakage")
    st.caption("  ·  ".join(tags))
    st.write(q["description"])

    if q["leak"]:
        st.error(f"**This column defines the outcome.** {q['leak_reason']} "
                 f"A model using it cannot give an early warning.")
    if q["owners"]:
        st.caption(f"Only recorded in the **{', '.join(q['owners'])}** study, so it is graded "
                   f"against those {q['ref_n']:,} students rather than the whole file.")

    c1, c2 = st.columns([1, 1])
    c1.markdown(f"**Possible values**  \n{q['range_text']}")
    if q["unlocks"]:
        c2.info(f"Answering this unlocks the **{q['unlocks']} · {EXP_NAME[q['unlocks']]}** model.")

    if q["bands"]:
        with st.expander("What counts as a good or bad value here?", expanded=True):
            st.caption("Worked out from your own dataset by comparing students who passed with "
                       "students who failed inside the same study — not a generic rule of thumb.")
            st.markdown(
                "<table style='width:100%;font-size:0.85rem'>"
                "<tr><th align='left'>Value</th><th align='left'>Interpretation</th>"
                "<th align='left'>Meaning in this cohort</th></tr>"
                + "".join(f"<tr><td><b>{b['range']}</b></td><td>{chip(b['status'])}</td>"
                          f"<td>{b['meaning']}</td></tr>" for b in q["bands"])
                + "</table>", unsafe_allow_html=True)
    elif q["contradicts"]:
        st.warning("In this data the relationship runs opposite to common sense, most likely "
                   "because of confounding, so it is left ungraded rather than producing "
                   "harmful advice.")
    else:
        st.caption("This field does not separate passing from failing students in its cohort, "
                   "so there is no good/bad banding for it.")
    return q


def widget_for(feat, key, current=None):
    """Right input control for the feature type, returning a RAW value."""
    sp = E.specs[feat]
    if sp["kind"] == "categorical":
        cur = current if current in sp["options"] else sp["default"]
        return st.selectbox("Your answer", sp["options"],
                            index=sp["options"].index(cur) if cur in sp["options"] else 0, key=key)
    if sp["kind"] == "ordinal":
        keys = sp["level_keys"]
        cur = int(current) if current is not None and int(current) in keys else int(sp["default"])
        return st.selectbox("Your answer", keys,
                            index=keys.index(cur) if cur in keys else 0,
                            format_func=lambda k: E.fmt(feat, k), key=key)
    lbl = f"Your answer ({sp['unit']})" if sp["unit"] else "Your answer"
    return st.number_input(lbl, min_value=float(sp["min"]), max_value=float(sp["max"]),
                           value=float(current if current is not None else sp["default"]),
                           step=float(sp["step"]), key=key)


def live_feedback(feat, value):
    try:
        a = E.assess(feat, E.encode({feat: value})[feat])
    except Exception:
        return
    extra = f" &nbsp;·&nbsp; ideal: **{a['ideal']}**" if a["ideal"] != "-" else ""
    st.markdown(f"{chip(a['status'])} &nbsp; {a['comment']}{extra}", unsafe_allow_html=True)


# ================================================================ results view
def render_results(results: dict, student_id: str, raw_answers: dict):
    if not results:
        st.warning("Not enough information yet to run any model. Answer at least the "
                   "very-early (E3) questions.")
        return
    ov = Engine.overall(results)
    head = results[ov["exp"]]

    # ------------------------------------------------------------- AI report
    report_data = ReportFormatter.build(ov, results)
    try:
        with st.spinner("Generating AI report…"):
            llm_markdown = llm.generate(report_data)
    except Exception as e:
        llm_markdown = None
        st.warning(f"AI report unavailable right now: {e}")

    #pdf = AIReportPDF()

    # Build AI PDF
    ai_pdf = None
    if llm_markdown:
        ai_pdf = ai_pdf_generator.build(
            student_name=student_name,
            prediction=ov["verdict"],
            confidence=ov["confidence"] * 100,
            markdown_report=llm_markdown
        )
    
    # ---------------------------------------------------------------- headline
    st.subheader("Overall assessment")
    k = st.columns(5)
    k[0].metric("Failure probability", pct_text(ov["fail_prob"]))
    k[1].metric("Pass probability", pct_text(ov["pass_prob"]))
    k[2].metric("Confidence", f"{ov['confidence']*100:.0f}%", ov["level"], delta_color="off")
    lo, hi = ov["interval"]
    k[3].metric("Likely range", f"{pct_text(lo)}–{pct_text(hi)}")
    with k[4]:
        st.caption("Risk level")
        st.markdown(risk_chip(ov["risk_level"], ov["risk_colour"]), unsafe_allow_html=True)
        st.caption(f"Intervention: **{ov['urgency']}**")
    st.caption(ov["note"])

    if head.get("leaks_used"):
        st.error(f"**Target leakage:** this model uses {', '.join(head['leaks_used'])}, which "
                 f"define the pass/fail outcome. Its accuracy is not a real forecast — retrain "
                 f"without these columns.")
    if len(results) > 1 and not ov["agree"]:
        st.warning(f"The horizons disagree. Later horizons see more information, so trust "
                   f"**{ov['exp']}** — but the disagreement itself signals a changing trajectory.")

    if llm_markdown:
        with st.expander("🤖 AI Generated Report", expanded=True):
            st.markdown(llm_markdown)

    # ---------------------------------------------------------- horizon chart
    st.subheader("Prediction results")
    cols = st.columns(3)
    for c, exp in zip(cols, ["E3", "E2", "E1"]):
        with c:
            if exp in results:
                r = results[exp]
                cf = r["confidence"]
                st.metric(f"{exp} · {EXP_NAME[exp]}", f"{pct_text(r['fail_prob'])} fail",
                          r["verdict"], delta_color="inverse")
                st.markdown(risk_chip(r["risk_level"], r["risk_colour"]), unsafe_allow_html=True)
                st.caption(f"pass {pct_text(r['pass_prob'])} · confidence {cf['level']} "
                           f"({cf['score']*100:.0f}%)")
            else:
                st.metric(f"{exp} · {EXP_NAME[exp]}", "—", "needs more data")

    runnable = [e for e in ["E3", "E2", "E1"] if e in results]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    probs = [results[e]["fail_prob"] * 100 for e in runnable]
    lo_ = [max(.1, (results[e]["fail_prob"] - results[e]["confidence"]["interval"][0]) * 100)
           for e in runnable]
    hi_ = [max(.1, (results[e]["confidence"]["interval"][1] - results[e]["fail_prob"]) * 100)
           for e in runnable]
    names = [f"{e}\n{EXP_NAME[e]}" for e in runnable]
    bars = ax.bar(names, probs, width=.5, color=[results[e]["risk_colour"] for e in runnable],
                  zorder=3)
    ax.errorbar(names, probs, yerr=[lo_, hi_], fmt="none", ecolor="#334155", elinewidth=1.2,
                capsize=6, zorder=4)
    ax.axhline(50, ls="--", lw=1, color="#94a3b8")
    ax.set_ylabel("Failure probability (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", color="#eef2f6", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for b, p in zip(bars, probs):
        ax.text(b.get_x() + b.get_width() / 2, p + 4, f"{p:.0f}%", ha="center", fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)

    # ------------------------------------------------------ per-horizon detail
    for exp in runnable:
        r = results[exp]
        with st.expander(f"{exp} · {EXP_NAME[exp]} — full breakdown",
                         expanded=(exp == runnable[0])):
            st.caption(EXP_WINDOW[exp])
            st.write(Engine.narrative(r))

            cf = r["confidence"]
            cc = st.columns(4)
            for col, (nm, v) in zip(cc, [("Decision margin", cf["margin"]),
                                         ("Model stability", cf["stability"]),
                                         ("Typical of cohort", cf["typicality"]),
                                         ("Inputs supplied", cf["completeness"])]):
                col.progress(float(np.clip(v, 0, 1)), text=f"{nm}: {v:.2f}")

            st.markdown("**SHAP explainability**")
            left, right = st.columns([3, 2])
            with left:
                plt.close("all")
                shap.plots.waterfall(r["explanation"], max_display=12, show=False)
                st.pyplot(plt.gcf())
                plt.close("all")
            with right:
                if r["risk"]:
                    st.markdown("**Top positive contributors** (toward failing)")
                    for n, s in r["risk"]:
                        st.write(f"🔺 {n}  ({s:+.2f})")
                if r["protective"]:
                    st.markdown("**Top negative contributors** (toward passing)")
                    for n, s in r["protective"]:
                        st.write(f"🟢 {n}  ({s:+.2f})")

            st.markdown("**Student profile analysis**")
            st.caption("Status compares this student with students from the same source study "
                       "who passed. Influence is how strongly the model used the answer here.")
            wide(st.dataframe, pd.DataFrame([{
                "Feature": f["label"] + ("  (background)" if not f["actionable"] else "")
                                      + ("  (LEAK)" if f["leak"] else ""),
                "Value": f["display"], "Ideal": f["ideal"], "Status": f["status"],
                "Influence": f["influence"], "Effect": f["effect"], "SHAP": round(f["shap"], 3),
                "Why": f["comment"],
            } for f in r["factors"]]), hide_index=True)

            g1, g2, g3 = st.columns(3)
            for col, title, items, empty in [
                    (g1, "✅ Strengths", r["strengths"], "Nothing above the passing benchmark."),
                    (g2, "⚠️ Watch", r["watch"], "No borderline factors."),
                    (g3, "🔴 Weaknesses", r["concerns"], "Nothing in the failing range.")]:
                with col:
                    st.markdown(f"**{title}**")
                    if not items:
                        st.caption(empty)
                    for f in items[:6]:
                        st.markdown(f"{chip(f['status'])} **{f['label']}** — {f['display']}",
                                    unsafe_allow_html=True)
                        st.caption(f["comment"])

            nr = [f["label"] for f in r["factors"] if f["status"] == "Not recorded"]
            if nr:
                st.caption(f"Not measured for this student (padding in the merged file, so "
                           f"excluded from grading and recommendations): {', '.join(nr)}.")
            if r["background"]:
                st.caption("Background factors raising risk (not actionable — these show who "
                           "needs more resourcing, never who deserves less effort): "
                           + ", ".join(r["background"]) + ".")

            st.markdown("**Recommendations — measured by re-running the model**")
            if not r["recommendations"]:
                st.info("No single controllable change moved this prediction. Escalate to a "
                        "human case review rather than a one-off intervention.")
            else:
                st.caption("Targets come from what passing students in the same study look "
                           "like. Apply in order — each figure assumes the ones above it.")
                wide(st.dataframe, pd.DataFrame([{
                    "#": x["rank"], "Change": x["label"], "Now": x["current"],
                    "Target": x["target"],
                    "Failure risk": f"{pct_text(x['from_prob'])} → {pct_text(x['to_prob'])}",
                    "On its own": x["solo_text"], "Status": x["status"],
                } for x in r["recommendations"]]), hide_index=True)
                plan = r["what_if"].get("plan")
                if plan:
                    st.success(f"Doing all {len(plan['steps'])} moves the failure probability "
                               f"from **{pct_text(plan['base_prob'])}** to "
                               f"**{pct_text(plan['new_prob'])}**.")

            a1, a2 = st.columns(2)
            with a1:
                if r["immediate"]:
                    st.markdown("**Immediate actions** (this week)")
                    for s in r["immediate"]:
                        st.write(f"- {s}")
            with a2:
                if r["suggestions"]:
                    st.markdown("**Suggestions** (longer term)")
                    for s in r["suggestions"]:
                        st.write(f"- {s}")

    # ------------------------------------------------------------- summaries
    st.subheader("Final summary")
    sm = Engine.audience_summaries(head, student_id)
    t1, t2, t3 = st.tabs(["👩‍🏫 Teacher", "👨‍👩‍👧 Parent", "🎓 Student"])
    for tab, key in [(t1, "teacher"), (t2, "parent"), (t3, "student")]:
        with tab:
            st.write(sm[key])

    # ------------------------------------------------------- what-if simulator
    with st.expander("🧪 What-if simulator — change inputs and re-score live"):
        exp = st.selectbox("Model", runnable, index=0, key="wi_exp")
        src = raw_answers.get("source")
        feats = [f for f in E.features[exp]
                 if E.specs[f]["actionable"] and not E.specs[f]["leak"]
                 and not (src and E.specs[f]["owners"]
                          and str(src) not in [str(o) for o in E.specs[f]["owners"]])]
        if not feats:
            st.caption("No controllable, measured inputs in this model for this student.")
        else:
            enc = E.encode(raw_answers)
            sim = {k: v for k, v in enc.items() if k != "__source__"}
            cols = st.columns(2)
            for i, f in enumerate(feats):
                sp = E.specs[f]
                with cols[i % 2]:
                    cur_raw = E.decode(f, enc[f])
                    v = widget_for(f, f"sim_{f}",
                                   current=cur_raw if sp["kind"] != "categorical" else str(cur_raw))
                    sim[f] = E.encode({f: v})[f]
            base_p = results[exp]["fail_prob"]
            new_p = float(E._probs([{k: sim[k] for k in E.features[exp]}], exp)[0])
            d1, d2 = st.columns(2)
            d1.metric("Original", pct_text(base_p, 1))
            d2.metric("Simulated", pct_text(new_p, 1), f"{(new_p-base_p)*100:+.1f} pts",
                      delta_color="inverse")

    # ------------------------------------------------------------------- PDF
    st.divider()
    with st.spinner("Building report…"):
        pdf = build_pdf(student_id, results, engine=E, raw_answers=raw_answers)
        st.download_button(
            label="📄 Download Detailed Report",
            data=pdf,
            file_name="Student_Detailed_Report.pdf",
            mime="application/pdf"
        )
        st.download_button(
            label="🤖 Download AI Summary",
            data=ai_pdf,
            file_name="Student_AI_Report.pdf",
            mime="application/pdf"
        )
    st.caption("The PDF covers the prediction, SHAP explainability, a graded profile of every "
               "feature, strengths, weaknesses, the measured action plan, intervention "
               "priority, teacher/parent/student summaries and a full appendix.")


# ================================================================ sidebar
st.title("🎓 Student Performance Risk Predictor")
mode = st.sidebar.radio("Input mode", ["Guided (one question at a time)", "Upload CSV / Excel"])
student_name = st.sidebar.text_input("Student name / ID", "Guided Student")
st.sidebar.caption("E3 = very early · E2 = early · E1 = late. Answer more to unlock later models.")
st.sidebar.divider()
st.sidebar.caption(f"Reference data: {len(E.raw):,} students · "
                   f"{(1-E.base_rate)*100:.0f}% passed")
if getattr(E, "owners", None) and "source" in E.raw.columns:
    with st.sidebar.expander("⚠️ Merged dataset"):
        st.caption("This file stacks several studies. Each one only fills its own columns, so "
                   "every benchmark is computed inside the study a column came from.")
        st.dataframe(E.raw["source"].value_counts().rename("rows"), width="stretch")
if E.leaks:
    with st.sidebar.expander("🚨 Target leakage"):
        for f, why in E.leaks.items():
            st.caption(f"**{f}** — {why}")


# ================================================================ guided mode
if mode.startswith("Guided"):
    ss = st.session_state
    ss.setdefault("step", 0)
    ss.setdefault("answers", {})
    order, total = E.wizard_order, len(E.wizard_order)
    step = ss.step

    unlocked = E.runnable(E.encode(ss.answers))
    st.progress(min(step, total) / total,
                text=f"Question {min(step+1, total)} of {total}  ·  Unlocked: "
                     + (", ".join(unlocked) if unlocked else "none yet"))

    if step < total:
        feat = order[step]
        question_panel(feat)
        val = widget_for(feat, f"w_{feat}", current=ss.answers.get(feat))
        live_feedback(feat, val)

        c1, c2, c3 = st.columns(3)
        if c1.button("◀ Back", disabled=(step == 0)):
            ss.answers[feat] = val
            ss.step -= 1
            st.rerun()
        if c2.button("Next ▶"):
            ss.answers[feat] = val
            ss.step += 1
            st.rerun()
        if unlocked and c3.button("✅ Predict now"):
            ss.answers[feat] = val
            ss.show = True
            st.rerun()
    else:
        st.success("All questions answered.")
        ss.show = True

    if ss.get("show"):
        st.divider()
        render_results(E.explain_all(E.encode(ss.answers)),
                       student_name or "Guided Student", ss.answers)
        if st.button("↺ Start over"):
            ss.step, ss.answers, ss.show = 0, {}, False
            st.rerun()

    with st.sidebar.expander("Answers so far"):
        if ss.answers:
            st.dataframe(pd.DataFrame([{"Question": label(k), "Answer": E.fmt(k, v)}
                                       for k, v in ss.answers.items()]), hide_index=True)
        else:
            st.caption("Nothing answered yet.")


# ================================================================ upload mode
else:
    st.write("Upload a CSV/Excel with student rows using the **raw values** "
             "(e.g. `gender = male`), matching your dataset's columns. If the file has a "
             "`source` column it will be used to tell real measurements from padding.")
    up = st.file_uploader("File", type=["csv", "xlsx", "xls"])
    if up:
        df = pd.read_csv(up) if up.name.endswith("csv") else pd.read_excel(up)
        st.write(f"Loaded **{len(df):,}** rows.")
        cap = st.number_input("Score how many rows?", 1, len(df), min(len(df), 200))

        with st.spinner("Scoring…"):
            summary, all_issues = [], []
            for idx, row in df.head(int(cap)).iterrows():
                iss = []
                enc = E.encode(row.to_dict(), issues=iss)
                all_issues += [f"row {idx} - {m}" for m in iss]
                rec = {"row": idx}
                if "source" in df.columns:
                    rec["source"] = row["source"]
                for e in E.runnable(enc):
                    d = E.explain(enc, e)
                    rec[f"{e}_fail%"] = round(d["fail_prob"] * 100, 1)
                    rec[f"{e}_risk"] = d["risk_level"]
                    rec[f"{e}_confidence"] = d["confidence"]["level"]
                summary.append(rec)
        sdf = pd.DataFrame(summary)
        wide(st.dataframe, sdf, hide_index=True)

        prob_cols = [c for c in sdf.columns if c.endswith("fail%")]
        if prob_cols:
            worst = prob_cols[-1]
            st.caption(f"Triage list sorted by `{worst}` — start at the top.")
            wide(st.dataframe, sdf.sort_values(worst, ascending=False).head(25), hide_index=True)

        if all_issues:
            with st.expander(f"⚠️ {len(all_issues)} value(s) could not be used"):
                for m in all_issues[:100]:
                    st.write(f"- {m}")

        pick = st.number_input("Inspect row #", 0, max(len(df) - 1, 0), 0)
        if st.button("Explain this student"):
            row = df.iloc[int(pick)].to_dict()
            render_results(E.explain_all(E.encode(row)), f"Student row {int(pick)}", row)