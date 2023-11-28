import os
from pathlib import Path

import logging
from aiera_gpt.config import LoggingSettings

ROOT_DIR = str(Path(__file__).parent.parent.absolute())

logging_settings = LoggingSettings()

logger = logging.getLogger("aiera_gpt")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging_settings.level)
