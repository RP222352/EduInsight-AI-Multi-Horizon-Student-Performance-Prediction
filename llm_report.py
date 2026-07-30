"""
llm_report.py

Converts structured prediction output into a professional
Markdown report using an Ollama LLM.

This module NEVER performs prediction.
It ONLY explains existing results.
"""

import json
import ollama

from llm_prompt import PromptLoader


class LLMReportGenerator:
    """
    Generates natural-language reports from structured
    prediction results.
    """

    def __init__(
        self,
        model="llama3:latest",
        template="report_prompt.txt"
    ):
        self.model = model

        loader = PromptLoader()
        self.system_prompt = loader.load(template)

    def generate(self, report_data: dict) -> str:
        """
        Generate Markdown report.

        If Ollama is unavailable,
        return a simple message instead of crashing.
        """

        json_data = json.dumps(
            report_data,
            indent=2,
            ensure_ascii=False,
            default=str
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": json_data,
                    },
                ],
            )

            return response["message"]["content"]

        except Exception as e:
            return (
                "# AI Report\n\n"
                "The AI-generated report could not be created.\n\n"
                f"Reason: {e}"
            )