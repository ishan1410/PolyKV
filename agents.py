import torch
from pool import SharedKVPool
from transformers.cache_utils import DynamicCache

class PooledAgent:
    """
    An agent that reads KV memory from a SharedKVPool
    instead of maintaining its own KV cache.

    The key experiment: do agents with SHARED compressed KV
    produce the same quality output as agents with their own
    full-precision KV? This class enables that measurement.
    """

    def __init__(self, agent_id: str, pool: SharedKVPool, model, tokenizer):
        self.agent_id = agent_id
        self.pool = pool
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, query: str, max_tokens: int = 200) -> str:
        """
        Generate a response using the shared compressed KV pool
        as the document context memory.
        """
        # Tokenize query
        input_ids = self.tokenizer.encode(query, return_tensors="pt")

        # Build DynamicCache layer by layer
        # Each layer's KV goes to the device that layer actually lives on
        cache = DynamicCache()
        for layer_idx in range(len(self.model.model.layers)):
            layer_device = (
                self.model.model.layers[layer_idx]
                .self_attn.q_proj.weight.device
            )
            k, v = self.pool.get_kv_for_layer(layer_idx)
            cache.update(
                k.to(self.model.dtype).to(layer_device),
                v.to(self.model.dtype).to(layer_device),
                layer_idx,
            )

        # Put input_ids on the same device as the first layer
        first_device = (
            self.model.model.layers[0].self_attn.q_proj.weight.device
        )
        input_ids = input_ids.to(first_device)

        seq_len = cache.get_seq_length()
        attention_mask = torch.ones(
            1, seq_len + input_ids.shape[1],
            dtype=torch.long, device=first_device
        )
        cache_position = torch.arange(
            seq_len, seq_len + input_ids.shape[1],
            device=first_device
        )

        # Generate with injected past_key_values
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=cache,
                max_new_tokens=max_tokens,
                do_sample=False,  # greedy for reproducibility
            )

        decoded = self.tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        return decoded
