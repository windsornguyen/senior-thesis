import torch
import torch.nn as nn

from packaging import version
from typing import Optional, Tuple, Union

from torch._utils import _get_available_device_type, _get_device_module
from thesis.utils.logger import logger


def get_device_info(
    return_type: str = "all",
) -> Union[str, torch.device, Tuple[str, torch.device], Tuple[str, torch.device, torch.device]]:
    """Get device information based on system availability.

    Args:
        return_type: What to return. Options:
            - "type": just the device type string (e.g. "cuda", "cpu")
            - "module": just the device module (e.g. torch.cuda)
            - "device": just the device instance (e.g. torch.device("cuda"))
            - "all": tuple of (type, module, device)
            - "type_device": tuple of (type, device)

    Returns:
        Device information based on return_type
    """
    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"  # Default to CUDA

    device_module = _get_device_module(device_type)
    if device_type == "cuda":
        device = torch.device("cuda")
    elif device_type == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if return_type == "type":
        return device_type
    elif return_type == "module":
        return device_module
    elif return_type == "device":
        return device
    elif return_type == "type_device":
        return device_type, device
    elif return_type == "all":
        return device_type, device_module, device
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def get_num_params(model: nn.Module, exclude_embeddings: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embeddings:
        num_params -= model.tok_emb.weight.numel()
    return num_params


def check_if_feature_in_pytorch(
    feature_name: str,
    pull_request: str | int,
    min_nightly_version: Optional[str] = None,
) -> bool:
    """
    Checks whether a specific PyTorch feature (introduced in a given pull request)
    is likely included in the installed PyTorch version. Logs a warning if it isn't.

    Args:
        feature_name (str):
            The name (or brief description) of the feature you want to check.
        pull_request (str | int):
            Either the pull request number (e.g., 12345) or a GitHub link
            (e.g., "https://github.com/pytorch/pytorch/pull/12345").
        min_nightly_version (Optional[str], optional):
            The minimum PyTorch nightly version that is known to include
            this feature. If the installed version is below this, a warning
            will be logged.

    Returns:
        bool:
            - True if it looks like PyTorch includes the given feature.
            - False if it might not include the feature (e.g., older version,
              or built from source without the PR merged).
    """
    # Convert a PR number to a GitHub link if needed
    if isinstance(pull_request, int) or not str(pull_request).startswith("http"):
        pull_request_link = f"https://github.com/pytorch/pytorch/pull/{pull_request}"
    else:
        pull_request_link = str(pull_request)

    # Attempt to parse the installed PyTorch version
    try:
        torch_version = version.parse(torch.__version__)
    except Exception as e:
        # If parsing fails, we cannot conclusively compare
        logger.warning(
            f"Unable to parse PyTorch version '{torch.__version__}': {e}. "
            "Cannot confirm if the feature is included."
        )
        return False

    # Check if PyTorch is built from source (often has '+git...' in its version)
    # or a local/build tag. If so, we cannot be sure the PR is included.
    if torch_version.local is not None:
        logger.warning(
            f"Detected that PyTorch '{torch.__version__}' is built from source. "
            f"Please ensure PR ({pull_request_link}) is merged into your local build "
            f"for correct '{feature_name}' functionality."
        )
        return False

    # If a minimum nightly version is specified, compare against the current version
    if min_nightly_version is not None:
        try:
            required_version = version.parse(min_nightly_version)
            if torch_version < required_version:
                logger.warning(
                    f"Detected that PyTorch version {torch_version.public} "
                    f"is older than {required_version.public}. "
                    f"Please upgrade to a newer version that includes the change "
                    f"in PR ({pull_request_link}) for correct '{feature_name}' functionality."
                )
                return False
        except Exception as e:
            logger.warning(
                f"Unable to parse the minimum required nightly version '{min_nightly_version}': {e}. "
                "Cannot confirm if the feature is included."
            )
            return False

    # If all checks passed, it's likely the feature is included
    return True
