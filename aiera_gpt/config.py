from pathlib import Path
from typing import NewType, List, Literal

from pydantic_settings import BaseSettings


class LoggingSettings(BaseSettings):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    class Config:
        env_prefix = "LOGGING_"

class OpenAISettings(BaseSettings):
    api_token: str
    org_id: str
    persist_files: bool = True
    try_allowance: int = 3
    try_pause: int = 15
    assistant_id: str = "asst_7GJLGrw0786VJ01rSiH1CVFv"

    class Config:
        env_prefix = "OPENAI_"


class AieraSettings(BaseSettings):
    api_key: str
    base_url: str = "https://premium.aiera.com/api/events"
    class Config:
        env_prefix = "AIERA_"


openai_settings = OpenAISettings()
aiera_settings = AieraSettings()
