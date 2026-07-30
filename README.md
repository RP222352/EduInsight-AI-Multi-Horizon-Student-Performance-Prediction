# Student Performance Prediction and Explainable Educational Decision Support System

An AI-assisted system that predicts a student's academic risk at multiple stages of a term, explains **why** each prediction was made, measures how reliable that prediction is, estimates the effect of possible interventions, and generates plain-language reports for teachers, parents, and students.

It combines Machine Learning, Explainable AI (SHAP), Counterfactual Analysis, Confidence Estimation, and a Large Language Model into a single educational decision-support pipeline.

---

## Table of Contents

- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Multi-Horizon Prediction & Model Selection](#multi-horizon-prediction--model-selection)
- [Explainability & Decision Support](#explainability--decision-support)
- [AI-Generated Reports](#ai-generated-reports)
- [Project Structure](#project-structure)
- [Module Reference](#module-reference)
- [Technologies Used](#technologies-used)
- [Dataset & Reproducibility](#dataset--reproducibility)
- [Key Features](#key-features)
- [Current Limitations](#current-limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Objectives

- Predict student performance at multiple stages of the academic term.
- Identify students who are academically at risk.
- Explain predictions using SHAP explainability.
- Estimate prediction confidence independently of prediction probability.
- Generate evidence-based, ranked intervention recommendations.
- Simulate "what-if" scenarios by modifying inputs interactively.
- Produce professional, audience-specific AI-generated reports.
- Support teachers, parents, and students in educational decision-making.

---

## System Architecture

```
                    Student Information
                             │
                             ▼
                  Data Preprocessing
                             │
                             ▼
                     Feature Encoding
                             │
                             ▼
                  CatBoost Classification
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
          Prediction              SHAP Explanation
                │                         │
                └────────────┬────────────┘
                             ▼
                 Confidence Estimation
                             │
                             ▼
                Counterfactual Recommendation
                             │
                             ▼
              Structured Explanation Dictionary
                             │
               ┌─────────────┴─────────────┐
               │                           │
               ▼                           ▼
      Detailed PDF Report          AI Narrative Report
```

---

## Multi-Horizon Prediction & Model Selection

The system predicts performance at three progressively later stages of the term, so risk assessments can be updated as more information becomes available.

| Horizon | Stage | Purpose |
|---|---|---|
| **E3** | Very early prediction | Uses only information available in the initial weeks. Enables early intervention and identifies students needing immediate support. |
| **E2** | Mid-term prediction | Incorporates additional academic information collected during the semester. Used to update risk assessment and measure improvement or deterioration. |
| **E1** | Late prediction | Uses the complete set of available academic information. Produces the final, most comprehensive prediction and recommendation set. |

**Models evaluated per horizon:** Logistic Regression, Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, KNN, SVM, XGBoost, LightGBM, and CatBoost — each under identical preprocessing and validation procedures, with performance comparison reports generated automatically for every stage.

**Deployed model: CatBoost**, across all three horizons. K-Nearest Neighbours (KNN) achieved the highest raw accuracy at the E2 and E3 stages during experimentation, but was not selected for deployment, because:

- SHAP explanations for KNN are computationally expensive and less intuitive.
- Counterfactual analysis is less stable under KNN.
- Real-time inference is slower on larger datasets.

CatBoost was chosen as the unified deployment model instead, because it offers:

- Strong predictive accuracy alongside native handling of categorical variables and missing values.
- Efficient, native TreeSHAP support for explainability.
- Stable probability estimates and reliable counterfactual recommendation generation.
- One consistent explainability framework across E1, E2, and E3 — rather than a different modeling and explanation approach per horizon.

This design deliberately balances predictive performance against interpretability, which matters more for an educational decision-support tool than maximizing raw accuracy alone.

---

## Explainability & Decision Support

**SHAP (SHapley Additive Explanations)** attributes each prediction back to individual features, showing:

- Which features increased or reduced failure probability, and by how much.
- Global feature importance across the model.
- Waterfall plots, top positive/negative contributors, and feature-wise explanations.

**Confidence Estimation** is calculated independently of the prediction probability itself, combining:

- Decision margin
- Prediction stability
- Cohort typicality
- Input completeness

Output is reported as **High / Moderate / Low** confidence, indicating how reliable a given prediction is — not just how strongly it leans one way.

**Student Profile Analysis** breaks down every feature individually — value, status, influence, and effect on the prediction — for example:

```
Attendance
  Value:   74%
  Status:  Needs Improvement
  Influence: High
  Effect:  Increases failure probability
```

**Counterfactual Recommendation Engine** generates recommendations by re-running the model with one feature modified at a time, then measuring the resulting risk reduction:

```
Increase attendance
  Failure probability: 42% → 25%
```

Recommendations are ranked by expected improvement rather than generated as generic advice.

**What-If Simulator** lets educators interactively modify student inputs and immediately see the recalculated prediction, confidence, SHAP explanation, and recommendations — useful for evaluating hypothetical interventions before they're implemented.

---

## AI-Generated Reports

An LLM (via Ollama, running Llama 3 locally) converts the structured output above into natural-language reports. The LLM **never predicts** — it only receives already-computed evidence (prediction, confidence, strengths, weaknesses, recommendations) and generates narrative text for it. Prediction logic always stays inside the ML engine.

The application produces two independent reports from the same underlying data:

| | Detailed Technical Report | AI Summary Report |
|---|---|---|
| Generated by | `pdf_report.py` | `ai_pdf_report.py` |
| Audience | Educators wanting full analytical detail | Quick reading for educators and parents |
| Contents | Prediction, SHAP analysis, confidence analysis, full student profile, recommendations, counterfactual analysis, teacher/parent/student summaries | Executive summary, key findings, student strengths, areas requiring attention, immediate actions, long-term recommendations, teacher/parent/student summaries |

---

## Project Structure

```
StudentPerformance/
│
├── app.py                    # Main Streamlit application
├── spp_engine.py              # Core prediction & explainability engine
├── pdf_report.py              # Detailed technical PDF report
├── ai_pdf_report.py            # AI summary PDF report
├── llm_report.py               # Ollama communication layer
├── llm_prompt.py                # External prompt template loader
├── report_formatter.py           # Structures engine output for the LLM
│
├── templates/
│   └── report_prompt.txt          # LLM system prompt
│
├── Models/
├── Results/                       # Generated after training (see below)
└── requirements.txt
```

---

## Module Reference

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI, student input, prediction execution, report generation and downloads |
| `spp_engine.py` | Feature encoding, prediction, confidence estimation, SHAP generation, counterfactual recommendations, what-if simulation — the complete prediction logic |
| `pdf_report.py` | Renders the detailed technical PDF report |
| `ai_pdf_report.py` | Renders the concise AI summary PDF report |
| `llm_report.py` | Loads the prompt, sends structured JSON to Ollama, receives the Markdown report |
| `llm_prompt.py` | Loads external prompt templates, kept separate from code for easier maintenance |
| `report_formatter.py` | Converts prediction results into a compact JSON representation for the LLM, forwarding only essential information |

---

## Technologies Used

- **Frontend:** Streamlit
- **Language:** Python
- **Machine Learning:** CatBoost
- **Explainability:** SHAP
- **Visualization:** Matplotlib
- **Data Processing:** Pandas, NumPy
- **LLM:** Ollama (Llama 3, local)
- **PDF Generation:** ReportLab

---

## Dataset & Reproducibility

The dataset is **not included** in this repository due to licensing, privacy, and distribution considerations. Researchers or reviewers who require it for academic validation may request access from the authors.

The `Results/` directory is likewise empty in this repository. It's populated automatically when the training pipeline runs, and typically contains, per horizon (E1/E2/E3):

```
Results/E1/
├── Best_Model.pkl, Feature_Names.pkl, StandardScaler.pkl, LabelEncoder.pkl
├── X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl
├── X_train_SMOTE.pkl, y_train_SMOTE.pkl
├── Confusion_Matrix.png, ROC_Curve.png
├── Classification_Report.csv, Model_Comparison.csv
└── Models/
    ├── Logistic_Regression.pkl, Decision_Tree.pkl, Random_Forest.pkl
    ├── Extra_Trees.pkl, AdaBoost.pkl, Gradient_Boosting.pkl
    ├── KNN.pkl, SVM.pkl, XGBoost.pkl, LightGBM.pkl, CatBoost.pkl
```
(E2/ and E3/ follow the same structure.)

Upon reasonable academic request, the authors can provide the dataset (subject to availability and permissions), trained models, serialized preprocessing objects, and full experimental/evaluation reports for reproduction of published results.

---

## Key Features

- Multi-horizon prediction (E1 / E2 / E3)
- Explainable AI via SHAP
- Independent prediction confidence estimation
- Counterfactual, ranked intervention recommendations
- Interactive what-if simulator
- AI-generated narrative reports
- Detailed technical PDF reports
- Interactive Streamlit dashboard

---

## Current Limitations

- The AI report summarizes structured evidence rather than performing its own evidence-based reasoning.
- Prediction quality depends on the quality of the training dataset.
- Local LLM output quality depends on the selected Ollama model.
- Intervention recommendations currently consider one feature modification at a time, not combined interventions.

---

## Future Improvements

- Evidence-grounded AI reasoning.
- Multi-feature intervention optimization.
- Retrieval-augmented educational knowledge.
- Longitudinal student progress tracking.
- Institutional dashboard.
- Teacher feedback integration.
- Confidence calibration using external validation.
- Multi-agent educational recommendation system.

---

## License

This project is intended for academic research and educational decision support. It should assist teachers and educational institutions and must not replace professional academic judgment.
