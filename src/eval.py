import json
from pathlib import Path

from src.agent import Meetings
from src.transcriber import Transcriber
from src.utils import formated_text

from sentence_transformers import SentenceTransformer
from jiwer import cer, wer

GROUND_TRUTH_PATH = Path("data/ground_truth.json")
AUDIO_DIR = Path("data/raw_audio")
REPORT_PATH = Path("data/eval_report.json")


class MeetingEvaluator:
    def __init__(self):
        print("Loading embeddings model..")
        self.embeddings_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("Embeddings model loaded.")

        self.transcriber = Transcriber()
        self.agent = Meetings()

    # --------------------------------
    # Text Similarity
    # --------------------------------
    def semantic_similarity(self, predicted: str, expected: str) -> float:
        if not expected or not predicted:
            return 0.0

        embeddings = self.embeddings_model.encode(
            [predicted, expected],
            normalize_embeddings=True,
        )

        score = float(embeddings[0] @ embeddings[1])

        return max(0.0, min(1.0, score))

    # --------------------------------
    # List comparison (key_points / decisions / open_questions)
    # Each expected item is matched against its best-scoring predicted item,
    # then the scores are averaged. This tolerates items being in a
    # different order or phrased differently.
    # --------------------------------
    def list_similarity(self, predicted: list[str], expected: list[str]) -> float:
        if not expected:
            # nothing expected: perfect score only if nothing was invented
            return 1.0 if not predicted else 0.0

        if not predicted:
            return 0.0

        scores = []
        for exp_item in expected:
            best = max(
                self.semantic_similarity(pred_item, exp_item)
                for pred_item in predicted
            )
            scores.append(best)

        return sum(scores) / len(scores)

    # --------------------------------
    # Action items comparison (task + responsible_person)
    # --------------------------------
    def action_items_similarity(self, predicted: list[dict], expected: list[dict]) -> dict:
        if not expected:
            return {
                "task_score": 1.0 if not predicted else 0.0,
                "person_score": 1.0 if not predicted else 0.0,
            }

        if not predicted:
            return {"task_score": 0.0, "person_score": 0.0}

        task_scores = []
        person_scores = []

        for exp_item in expected:
            exp_task = exp_item.get("task", "")
            exp_person = (exp_item.get("responsible_person") or "").strip().lower()

            # find the predicted item whose task best matches this expected task
            best_pred = max(
                predicted,
                key=lambda p: self.semantic_similarity(p.get("task", ""), exp_task),
            )

            task_scores.append(
                self.semantic_similarity(best_pred.get("task", ""), exp_task)
            )

            pred_person = (best_pred.get("responsible_person") or "").strip().lower()
            person_scores.append(1.0 if pred_person == exp_person else 0.0)

        return {
            "task_score": sum(task_scores) / len(task_scores),
            "person_score": sum(person_scores) / len(person_scores),
        }

    # -------------------------------
    # Transcription evaluation
    # -------------------------------
    def evaluate_transcription(self, expected: str, predicted: str) -> dict:
        if not predicted or not expected:
            return {"wer": 1.0, "cer": 1.0}

        predicted = predicted.strip()
        expected = expected.strip()

        return {
            "wer": wer(expected, predicted),
            "cer": cer(expected, predicted),
        }

    # -------------------------------
    # Full pipeline evaluation for one meeting
    # -------------------------------
    def evaluate_meeting(self, meeting_id: str, ground_truth: dict, audio_path: Path) -> dict:
        print(f"\n=== Evaluating {meeting_id} ===")

        # Audio -> transcript
        predicted_transcript = self.transcriber.transcribe(audio_path)
        transcription_scores = self.evaluate_transcription(
            expected=ground_truth["reference_transcript"],
            predicted=predicted_transcript,
        )

        # Transcript -> meeting notes
        cleaned_transcript = formated_text(predicted_transcript)
        predicted_notes = self.agent.analyze_transcript(cleaned_transcript)

        summary_score = self.semantic_similarity(
            predicted_notes.get("summary", ""), ground_truth.get("summary", "")
        )
        key_points_score = self.list_similarity(
            predicted_notes.get("key_points", []), ground_truth.get("key_points", [])
        )
        decisions_score = self.list_similarity(
            predicted_notes.get("decisions", []), ground_truth.get("decisions", [])
        )
        open_questions_score = self.list_similarity(
            predicted_notes.get("open_questions", []),
            ground_truth.get("open_questions", []),
        )
        action_items_scores = self.action_items_similarity(
            predicted_notes.get("action_items", []),
            ground_truth.get("action_items", []),
        )

        return {
            "transcription": transcription_scores,
            "summary_similarity": summary_score,
            "key_points_similarity": key_points_score,
            "decisions_similarity": decisions_score,
            "open_questions_similarity": open_questions_score,
            "action_items_task_similarity": action_items_scores["task_score"],
            "action_items_person_accuracy": action_items_scores["person_score"],
        }


def find_audio_file(meeting_id: str) -> Path | None:
    matches = list(AUDIO_DIR.glob(f"{meeting_id}.*"))
    return matches[0] if matches else None


def main():
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Ground truth file not found: {GROUND_TRUTH_PATH}")

    ground_truth_data = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))

    evaluator = MeetingEvaluator()

    report = {}

    for meeting_id, ground_truth in ground_truth_data.items():
        audio_path = find_audio_file(meeting_id)

        if audio_path is None:
            print(f"Skipping {meeting_id}: no matching audio file in {AUDIO_DIR}")
            continue

        report[meeting_id] = evaluator.evaluate_meeting(
            meeting_id, ground_truth, audio_path
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nEvaluation report saved to: {REPORT_PATH}")

    print("\n------ SUMMARY ------")
    for meeting_id, scores in report.items():
        print(f"\n{meeting_id}:")
        for key, value in scores.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()