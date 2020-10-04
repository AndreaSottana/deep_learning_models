import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def set_hardware_acceleration(default: Optional[str] = None) -> torch.device:
    """
    Helper function to set your device. If you don't specify a device argument, it will default to GPU if one is
    available, else it will use a CPU.
    :param default: can be one of cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu. Default: None
    :return: device: the torch.device to be used for training.
    """
    if default is not None:
        device = torch.device(default)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.warning(
                f"There are {torch.cuda.device_count()} GPUs available. Using the {torch.cuda.get_device_name()} GPU."
            )
        else:
            device = torch.device("cpu")
            logger.warning("No GPUs available, using CPU instead.")
    return device
