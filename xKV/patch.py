from typing import Dict, Optional
from loguru import logger

from transformers.cache_utils import Cache, DynamicCache
from .attn_patch.mistral import enable_mistral_xKV_eval
from .attn_patch.llama import enable_llama_xKV_eval
from .attn_patch.qwen import enable_qwen_xKV_eval
from .attn_patch.deepseek_v2 import enable_deepseek_v2_xKV_eval
from .customized_cache import method_to_cache_obj
from .configurations import xKVConfig

def prepare_cache(method: str, config):
    cache_obj: Cache = method_to_cache_obj.get(method, None)

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs: Dict, *args, **kwargs
    ) -> bool:
        """
        Prepares the cache for generation (if applicable), given `generate`'s paramaterization. If a cache is
        instantiated, writes it to `model_kwargs`, under the name expected by the model.
        """
        #config.num_layers = self.config.num_hidden_layers
        if cache_obj is None:
            model_kwargs["past_key_values"] = DynamicCache()
        else:
            model_kwargs["past_key_values"] = cache_obj(config)

    return _prepare_cache_for_generation



class KVCompress:
    """
    A helper to patch your model with a FakeLayerMergingCache 
    using a user-provided or YAML-loaded xKVConfig.
    """

    def __init__(self, xKV_config: Optional[xKVConfig] = None, yaml_path: Optional[str] = None):
        """
        If xKV_config is given, use that.
        Else if yaml_path is given, load from that YAML file.
        Otherwise, raise an error or set a default fallback.
        """
        if xKV_config is not None:
            self.config = xKV_config
        elif yaml_path is not None:
            self.config = xKVConfig.from_yaml(yaml_path)
        else:
            raise ValueError("Must provide either xKV_config or yaml_path.")
    
    def __call__(self, model):
        return self.enable_xKV_patch(model)
    
    def enable_xKV_patch(self, model):
        logger.info("Enabling xKV patch for model: {}".format(model.config.architectures))
        if "mistral" in model.config.model_type:
            enable_mistral_xKV_eval(model)
        elif "llama" in model.config.model_type:
            enable_llama_xKV_eval(model)
        elif "qwen" in model.config.model_type:
            enable_qwen_xKV_eval(model)    
        elif "deepseek_v2" in model.config.model_type:
            enable_deepseek_v2_xKV_eval(model)
        else:
            raise ValueError("Model type not supported for xKV patch: {}".format(model.config.model_type))
        
        model.kv_compress_config = self.config
        prepare_cache_fn = prepare_cache("xKV", self.config)
        model._prepare_cache_for_generation = prepare_cache_fn.__get__(
            model, model.__class__
        )
        
        return model