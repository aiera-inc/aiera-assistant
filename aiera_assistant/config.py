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
        api_key (str): The API key for accessing OpenAI.
        org_id (str): The organization ID for the user.
        persist_files (bool, optional): Whether to persist files. Defaults to True.
        try_allowance (int, optional): The number of allowed tries. Defaults to 3.
        try_pause (int, optional): The pause duration between tries. Defaults to 15.
        assistant_id (str, optional): The ID of the assistant.

    Configurations:
        env_prefix (str): The prefix for environment variables.
    """    
    api_key: str
    org_id: str
    persist_files: bool = True
    try_allowance: int = 3
    try_pause: int = 15
    assistant_id: str 

    class Config:
        env_prefix = "OPENAI_"


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

logging_settings = LoggingSettings()

logger = logging.getLogger("aiera_assistant")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging_settings.level)
