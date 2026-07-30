"""
llm_prompt.py

Loads the LLM system prompt from an external text file.
Keeping prompts outside the code makes them easier to
modify without changing Python files.
"""

from pathlib import Path


class PromptLoader:
    """Loads prompt templates."""

    def __init__(self, template_dir="templates"):
        self.template_dir = Path(template_dir)

    def load(self, filename="report_prompt.txt"):
        """
        Load a prompt template.

        Parameters
        ----------
        filename : str
            Prompt file inside templates/

        Returns
        -------
        str
            Prompt text
        """
        path = self.template_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {path}"
            )

        return path.read_text(encoding="utf-8")