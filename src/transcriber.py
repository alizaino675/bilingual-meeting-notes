from src.config import settings

from pathlib import Path
from faster_whisper import WhisperModel, download_model

class Transcriber:
    def __init__(self, model_size : str | None = None, device="cpu", compute_type = "int8"):
        model_size = model_size or settings.whisper_model
        print(f"Model size: {model_size}")

        model_path = download_model(model_size)

        self.model = WhisperModel(
            model_size_or_path=model_path,
            device=device,
            compute_type=compute_type
        )

        print ("Model has been loaded")

    def transcribe(self, audio_path):
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio File Not Found: {path}")

        segments, info = self.model.transcribe(
            str(path),
            beam_size=5,
            task="transcribe",
            vad_filter=True

        )

        transcript = " ".join(segment.text.strip() for segment in segments)

        return transcript.strip()


