import re


def clean_text(text: str):
    """Cleaning text before sending it to LLMs"""

    if not text:
        return ""

    # removing extra spaces
    text = re.sub(r"\s+", " ", text)

    # removing space before punctuation
    text = re.sub(r'\s+([,.!?،。！？])', r'\1', text)

    text = text.strip()

    return text


def formated_text(text: str):

    text = clean_text(text=text)

    if not text:
        return ""

    return f"Meeting transcripts: \n\n {text}"