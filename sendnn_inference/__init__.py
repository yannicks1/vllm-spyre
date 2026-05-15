import importlib.metadata
import json
from logging.config import dictConfig
from typing import Any

from vllm.envs import VLLM_CONFIGURE_LOGGING, VLLM_LOGGING_CONFIG_PATH
from vllm.logger import DEFAULT_LOGGING_CONFIG

__version__ = importlib.metadata.version("sendnn_inference")


# DEV HACK: granite41 SWA experimental support. Lets `vllm serve granite41`
# skip the HF Hub lookup and use the FMS Granite41Config defaults verbatim, so the
# plugin can drive the FMS-native granite41 model with random weights.
def _granite41_dev_hack():
    from transformers import AutoConfig, PretrainedConfig

    try:
        from fms.models.granite41 import Granite41Config
    except ImportError as e:
        raise ImportError(
            "granite41 dev hack requires fms.models.granite41 — make sure the "
            "foundation-model-stack submodule is checked out on the swa_granite branch"
        ) from e

    fms_cfg = Granite41Config()
    granite41_dict = {
        "model_type": "granite41",
        # vLLM rejects unknown architectures — borrow GraniteForCausalLM (registered in
        # vllm). Plugin dispatch still keys off model_type="granite41".
        "architectures": ["GraniteForCausalLM"],
        "num_hidden_layers": fms_cfg.nlayers,
        "hidden_size": fms_cfg.emb_dim,
        "num_attention_heads": fms_cfg.nheads,
        "num_key_value_heads": fms_cfg.kvheads,
        "head_dim": fms_cfg.head_dim,
        "vocab_size": fms_cfg.src_vocab_size,
        "max_position_embeddings": fms_cfg.max_expected_seq_len,
        "rope_theta": fms_cfg.rope_theta,
        "tie_word_embeddings": fms_cfg.tie_heads,
        "torch_dtype": "float16",
    }

    _orig_from_pretrained = AutoConfig.from_pretrained
    _orig_get_config_dict = PretrainedConfig.get_config_dict

    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if pretrained_model_name_or_path == "granite41":
            return PretrainedConfig(**granite41_dict)
        return _orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)

    @classmethod
    def _patched_get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        if pretrained_model_name_or_path == "granite41":
            return dict(granite41_dict), {}
        return _orig_get_config_dict.__func__(cls, pretrained_model_name_or_path, **kwargs)

    AutoConfig.from_pretrained = _patched_from_pretrained
    PretrainedConfig.get_config_dict = _patched_get_config_dict

    # vLLM's get_config also performs a config.json existence check that bypasses
    # AutoConfig — patch it too. Plus several siblings that ModelConfig.__post_init__
    # calls and that would all 404 for our synthetic id.
    import vllm.transformers_utils.config as _vllm_cfg

    _orig_get_config = _vllm_cfg.get_config

    def _patched_get_config(model, *args, **kwargs):
        if model == "granite41":
            return PretrainedConfig(**granite41_dict)
        return _orig_get_config(model, *args, **kwargs)

    _vllm_cfg.get_config = _patched_get_config

    _orig_get_image = _vllm_cfg.get_hf_image_processor_config

    def _patched_get_image(model, *args, **kwargs):
        if model == "granite41":
            return {}
        return _orig_get_image(model, *args, **kwargs)

    _vllm_cfg.get_hf_image_processor_config = _patched_get_image

    # ModelConfig.__post_init__ also probes the repo for sentence-transformer
    # / pooling / multimodal markers via file_or_path_exists. Short-circuit it.
    import vllm.transformers_utils.repo_utils as _repo

    _orig_file_exists = _repo.file_or_path_exists

    def _patched_file_exists(model, *args, **kwargs):
        if model == "granite41":
            return False
        return _orig_file_exists(model, *args, **kwargs)

    _repo.file_or_path_exists = _patched_file_exists
    _vllm_cfg.file_or_path_exists = _patched_file_exists


_granite41_dev_hack()


def register():
    """Register the Spyre platform."""
    return "sendnn_inference.platform.SpyrePlatform"


def _init_logging():
    """Setup logging, extending from the vLLM logging config"""
    config: dict[str, Any] = {}

    if VLLM_CONFIGURE_LOGGING:
        config = {**DEFAULT_LOGGING_CONFIG}

    if VLLM_LOGGING_CONFIG_PATH:
        # Error checks must be done already in vllm.logger.py
        with open(VLLM_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            config = json.loads(file.read())

    if VLLM_CONFIGURE_LOGGING:
        # Copy the vLLM logging configurations for our package
        if "sendnn_inference" not in config["formatters"]:
            if "vllm" in config["formatters"]:
                config["formatters"]["sendnn_inference"] = config["formatters"]["vllm"]
            else:
                config["formatters"]["sendnn_inference"] = DEFAULT_LOGGING_CONFIG["formatters"][
                    "vllm"
                ]

        if "sendnn_inference" not in config["handlers"]:
            if "vllm" in config["handlers"]:
                handler_config = config["handlers"]["vllm"]
            else:
                handler_config = DEFAULT_LOGGING_CONFIG["handlers"]["vllm"]
            handler_config["formatter"] = "sendnn_inference"
            config["handlers"]["sendnn_inference"] = handler_config

        if "sendnn_inference" not in config["loggers"]:
            if "vllm" in config["loggers"]:
                logger_config = config["loggers"]["vllm"]
            else:
                logger_config = DEFAULT_LOGGING_CONFIG["loggers"]["vllm"]
            logger_config["handlers"] = ["sendnn_inference"]
            config["loggers"]["sendnn_inference"] = logger_config

    dictConfig(config)


_init_logging()
