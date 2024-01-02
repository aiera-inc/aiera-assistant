from pathlib import Path
from typing import NewType, List, Literal
import logging
from pydantic_settings import BaseSettings

from aiera_assistant.__init__ import ROOT_DIR

class LoggingSettings(BaseSettings):
    """
    A class representing logging settings.

    Attributes:
        level (Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']): The log level for the application.
    
    Configurations:
        env_prefix (str): The prefix for environment variables used to configure the logging settings.
    """    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    class Config:
        env_prefix = "LOGGING_"

class OpenAISettings(BaseSettings):
    """
    A class representing OpenAI settings.

    Attributes:
        api_token (str): The API token for accessing OpenAI.
        org_id (str): The organization ID for the user.
        persist_files (bool, optional): Whether to persist files. Defaults to True.
        try_allowance (int, optional): The number of allowed tries. Defaults to 3.
        try_pause (int, optional): The pause duration between tries. Defaults to 15.
        assistant_id (str, optional): The ID of the assistant. Defaults to 'asst_7GJLGrw0786VJ01rSiH1CVFv'.

    Configurations:
        env_prefix (str): The prefix for environment variables.
    """    
    api_token: str
    org_id: str
    persist_files: bool = True
    try_allowance: int = 3
    try_pause: int = 15
    assistant_id: str = "asst_7GJLGrw0786VJ01rSiH1CVFv"

    class Config:
        env_prefix = "OPENAI_"


class DBSettings(BaseSettings):
    """
    A class representing the database settings.

    Attributes:
        db_path (str): The path to the database file.

    Config:
        env_prefix (str): The prefix for environment variables used to override settings.
    """    
    db_path: str = f"{ROOT_DIR}/aiera_gpt/db/companies.db"

    class Config:
        env_prefix = "DB_"



class AieraSettings(BaseSettings):
    """
    A class representing Aiera settings.

    Attributes:
        api_key (str): The API key for Aiera.
        base_url (str, optional): The base URL for the Aiera API. Default is 'https://premium.aiera.com/api/'.

    Config:
        env_prefix (str): The environment variable prefix for Aiera settings.
    """    
    api_key: str
    base_url: str = "https://premium.aiera.com/api/"
    class Config:
        env_prefix = "AIERA_"


openai_settings = OpenAISettings()
aiera_settings = AieraSettings()
db_settings = DBSettings()

logging_settings = LoggingSettings()

logger = logging.getLogger("aiera_assistant")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging_settings.level)
