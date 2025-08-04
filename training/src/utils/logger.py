"""
Logging configuration and common logger object.
"""

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
