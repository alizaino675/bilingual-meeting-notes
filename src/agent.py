import json

from ollama import chat
from src.config import settings


class Meetings:
    def __init__(self):

        self.model = settings.llm_model

    def analyze_transcript(self, transcript: str):
        if not transcript or not transcript.strip():
            raise ValueError("Transcript Can Not Be Empty.")

        system_prompt = """
        You are an expert bilingual meeting assistant.

        You analyze meeting transcripts that may contain:
        - Arabic
        - English
        - A mixture of Arabic and English

        Extract the important information accurately.

        Return ONLY valid JSON with exactly this structure:

        {
            "summary": "A concise summary of the meeting",
            "key_points": [],
            "decisions": [],
            "action_items": [],
            "open_questions": []
        }

        Rules:

        1. Do not invent information.
        2. If information is not present, return an empty list.
        3. Preserve names, dates, numbers, deadlines, and responsibilities.
        4. Action items should identify the task and responsible person when available.
        5. Keep the output language consistent with the dominant language.
        6. Preserve important English business and technical terms.
        7. Do not add information that does not exist in the transcript.
        8. Return ONLY JSON. Do not use markdown.
        """

        user_prompt = f"""
        Analyze the following meeting transcript:

        --- TRANSCRIPT ---

        {transcript}

        --- END TRANSCRIPT ---
        """

        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0
            },
        )

        content = response.message.content

        if not content:
            raise ValueError(f"Model '{self.model}' returned an empty response.")

        try:
            return json.loads(content)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Model '{self.model}' returned invalid JSON:\n{content}"
            ) from exc