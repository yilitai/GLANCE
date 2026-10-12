import logging

from glancevae._model import GLANCE
from glancevae._settings import settings
from glancevae.utils.seed import seed_everything

# Standard library convention for packages: attach a NullHandler to the
# root package logger so that log records are silently discarded when the
# user has not configured logging themselves.  GLANCESettings immediately
# replaces this with the appropriate handler on instantiation.
logging.getLogger("glancevae").addHandler(logging.NullHandler())

__all__ = ["GLANCE", "settings", "seed_everything"]
