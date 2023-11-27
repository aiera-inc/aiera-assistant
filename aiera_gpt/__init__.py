from aiera.shared_services.env import Environment, Service
from aiera.shared_services.injection import context
import os

############################################################
# THIS ENVIRONMENT SETUP MUST BE BEFORE OTHER IMPORTS
############################################################

context.bind_to_instance(Environment, os.environ.get("ENVIRONMENT", "local"))
context.bind_to_instance(Service, "aiera-gpt")

