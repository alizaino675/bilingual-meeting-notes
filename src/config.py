from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Not currently used (the project calls a local Ollama model),
    # kept optional in case a Gemini-based provider is added later.
    gemini_api_key: str | None = None

    llm_model: str = "qwen2.5:latest"

    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()