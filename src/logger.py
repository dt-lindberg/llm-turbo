"""Sets up a logger that writes to stdout and outputs/<name>.log."""

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    os.makedirs("outputs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"outputs/{name}.log"),
        ],
    )
    return logging.getLogger()
