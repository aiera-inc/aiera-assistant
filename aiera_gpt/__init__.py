
from . import _version
__version__ = _version.get_versions()['version']

import os
import sys
from pathlib import Path

from aiera.shared_services.env import Environment, Service
from aiera.shared_services.injection import context

ROOT_DIR = str(Path(__file__).parent.parent.absolute())

############################################################
# THIS ENVIRONMENT SETUP MUST BE BEFORE OTHER IMPORTS
############################################################

context.bind_to_instance(Environment, os.environ.get("ENVIRONMENT", "local"))
context.bind_to_instance(Service, "aiera-ds")


from aiera_gpt.config import initialize_context
initialize_context()
