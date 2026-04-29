from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    voyage_api_key: str
    llm_model_name: str = "claude-sonnet-4-6"
    embedding_model: str = "voyage-3"
    chroma_persist_dir: str = "./chroma_db"
    full_docs_store_dir: str = "./full_docs_store"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
