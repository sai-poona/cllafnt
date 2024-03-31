import logging
import subprocess
from typing import List

import torch


def get_num_gpus():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 0


def run_with_error_handling(command: List[str], shell=False):
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess script failed with return code: {e.returncode}")
        raise RuntimeError(e)