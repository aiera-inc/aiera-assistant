from pathlib import Path
from typing import NewType, List

from pydantic import BaseSettings



class OpenAISettings(BaseSettings):
    api_token: str
    org_id: str
    try_allowance: int = 3
    try_pause: int = 15
    assistant_id: str = "asst_7GJLGrw0786VJ01rSiH1CVFv"

    class Config:
        env_prefix = "OPENAI_"


class DatabaseSettings(BaseSettings):
    read_url: str
    write_url: str = None

    class Config:
        env_prefix = "DATABASE_"


class AieraSettings(BaseSettings):
    api_key: str
    class Config:
        env_prefix = "AIERA_"


openai_settings = OpenAISettings()

database_settings = DatabaseSettings()

aiera_settings = AieraSettings()
