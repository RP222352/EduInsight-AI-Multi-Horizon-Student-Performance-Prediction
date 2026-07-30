"""
report_formatter.py

Converts Engine output into a compact JSON
for the LLM.

Only sends information that is useful for
writing the report.
"""

from typing import Dict


class ReportFormatter:

    @staticmethod
    def build(overall: Dict, results: Dict) -> Dict:
        """
        Build compact report dictionary.

        Parameters
        ----------
        overall : dict
            Engine.overall()

        results : dict
            Engine.explain_all()

        Returns
        -------
        dict
        """

        exp = overall["exp"]
        result = results[exp]

        strengths = [x["label"] for x in result["strengths"][:5]]

        # An empty strengths list does NOT mean the student is doing badly -
        # it means no single answer individually cleared the "clearly passing"
        # bar for its own feature, even though the combined prediction can
        # still be strongly positive. Precompute the right framing here
        # instead of asking a small local model to infer it correctly.
        if strengths:
            strengths_note = None
        elif overall["verdict"] != "AT RISK":
            strengths_note = (
                "No single answer individually stands out, but the overall "
                "combination of answers is solidly positive and supports the "
                "on-track prediction."
            )
        else:
            strengths_note = (
                "No factor currently performs above the passing benchmark "
                "for its own comparison group."
            )

        report = {

            "prediction": {
                "verdict": overall["verdict"],
                "risk_level": overall["risk_level"],
                "failure_probability": round(
                    overall["fail_prob"] * 100,
                    1
                ),
                "passing_probability": round(
                    overall["pass_prob"] * 100,
                    1
                ),
                "confidence": {
                    "score": round(
                        overall["confidence"] * 100,
                        1
                    ),
                    "level": overall["level"],
                    "note": overall["note"]
                }
            },

            "strengths": strengths,
            "strengths_note": strengths_note,

            "concerns": [
                x["label"]
                for x in result["concerns"][:5]
            ],

            "watch_items": [
                x["label"]
                for x in result["watch"][:5]
            ],

            "critical_items": [
                x["label"]
                for x in result["critical"]
            ],

            "recommendations": [
                r["action"]
                for r in result["recommendations"][:5]
            ],

            "immediate_actions":
                result["immediate"],

            "long_term_actions":
                result["suggestions"],

            "prediction_window":
                result["window"]

        }

        return report