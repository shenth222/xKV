from .fake_layer_merge_dynamic_cache import FakeLayerMergingCache
from transformers.cache_utils import DynamicCache

method_to_cache_obj = {
    "xKV": FakeLayerMergingCache,
}