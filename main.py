import argparse
import sys
from pathlib import Path

from src.agent import Meetings
from src.transcriber import Transcriber
from src.utils import formated_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe a meeting audio file and generate bilingual meeting notes."
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the audio file to transcribe (e.g. data/raw_audio/meeting.mp3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    audio_path = Path(args.audio_file)

    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Audio → Transcript
    transcriber = Transcriber()

    transcript = transcriber.transcribe(audio_path)

    # Clean transcript
    transcript = formated_text(transcript)

    # Transcript → Meeting Notes
    agent = Meetings()

    result = agent.analyze_transcript(transcript)

    print("\n------ MEETING NOTES ------\n")

    print("Summary:")
    print(result["summary"])

    print("\nKey Points:")
    for point in result["key_points"]:
        print(f"- {point}")

    print("\nDecisions:")
    for decision in result["decisions"]:
        print(f"- {decision}")

    print("\nAction Items:")
    for item in result["action_items"]:
        task = item.get("task", "")
        person = item.get("responsible_person", "Not specified")
        print(f"- {task} (Responsible: {person})")

    print("\nOpen Questions:")
    for question in result["open_questions"]:
        print(f"- {question}")


if __name__ == "__main__":
    main()