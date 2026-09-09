````markdown
# Bilingual Meeting Notes

An AI-powered meeting intelligence system that converts audio recordings into structured meeting notes.

The system supports bilingual meeting audio and uses speech recognition and a local LLM to automatically extract:

- Meeting summary
- Key points
- Decisions
- Action items
- Open questions

It also includes an evaluation pipeline to measure transcription and meeting-analysis quality.

---

## Architecture

```text
Audio File
    │
    ▼
 Whisper
    │
    ▼
Transcript
    │
    ▼
 Qwen
    │
    ▼
Meeting Analysis
    │
    ├── Summary
    ├── Key Points
    ├── Decisions
    ├── Action Items
    └── Open Questions
    │
    ▼
Evaluation
    │
    ├── WER
    ├── CER
    └── Semantic Similarity
````

---

## Features

### Speech-to-Text

Uses Whisper to transcribe meeting recordings.

### Meeting Analysis

A local Qwen model analyzes the transcript and generates structured meeting notes.

### Structured Output

The system extracts:

* Summary
* Key points
* Decisions
* Action items
* Responsible person
* Open questions

### Evaluation

The project includes an evaluation pipeline using:

* Word Error Rate (WER)
* Character Error Rate (CER)
* Semantic similarity
* Key point similarity
* Decision similarity
* Action item similarity
* Responsible-person accuracy

---

## Technologies

* Python
* Whisper
* Qwen
* Ollama
* Sentence Transformers
* JiWER
* RapidFuzz
* PyTorch

---

## Project Structure

```text
bilingual-meeting-notes/
│
├── data/
│   ├── ground_truth.json
│   └── eval_report.json
│
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── eval.py
│   ├── transcriber.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/bilingual-meeting-notes.git
cd bilingual-meeting-notes
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Ollama Setup

Install Ollama and make sure the Qwen model is available locally.

For example:

```bash
ollama pull qwen2.5:3b
```

Start Ollama:

```bash
ollama serve
```

---

## Usage

Run the meeting analysis:

```bash
python main.py data/raw_audio/meeting_01.mp3
```

The system will:

1. Transcribe the audio.
2. Process the transcript.
3. Analyze the meeting using Qwen.
4. Generate structured meeting notes.

Example output:

```text
------ MEETING NOTES ------

Summary:
The meeting discussed chronic absenteeism, particularly on Fridays,
and proposed solutions such as a pancake breakfast and health tips.

Key Points:
- Chronic absenteeism, especially on Fridays
- Proposed a pancake breakfast to encourage attendance
- Health tips for flu season

Decisions:
- Plan a pancake breakfast next week
- Create posters with health tips for flu season

Action Items:
- Plan and organize a pancake breakfast
- Create and distribute posters with health tips
- Talk to John Smith
- Look for community resources for John's family
```

---

## Evaluation

The project includes an evaluation script that compares the generated results against manually prepared ground-truth data.

Run:

```bash
python -m src.eval
```

Example evaluation:

```text
meeting_01:

WER: 6.2%
CER: 4.26%

Summary Similarity: 94.66%
Key Points Similarity: 93.19%
Decisions Similarity: 100%
Open Questions Similarity: 100%

Action Items Task Similarity: 100%
Action Items Person Accuracy: 100%
```

The evaluation report is saved to:

```text
data/eval_report.json
```

---

## Evaluation Methodology

### Transcription

The generated transcript is compared with a reference transcript using:

* WER — Word Error Rate
* CER — Character Error Rate

Lower values indicate better transcription quality.

### Meeting Intelligence

Generated meeting information is compared against manually created ground truth using semantic similarity and task-specific matching.

This evaluates whether the system correctly captures the meaning and important information from the meeting rather than only comparing exact wording.

---

## Future Improvements

* Support longer meeting recordings
* Speaker diarization
* Better responsible-person extraction
* Improved multilingual evaluation
* Web-based user interface
* Automatic meeting report export
* Database storage for meeting history
* Real-time meeting transcription

---

## License

This project is intended for educational and portfolio purposes.
