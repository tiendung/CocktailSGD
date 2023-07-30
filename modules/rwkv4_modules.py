# Rút gọn từ https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/src/model.py
import torch, os, math, gc, types
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .rwkv4_timemix_channelmix import RWKV_TimeMix, RWKV_ChannelMix, init_weight

class GPTBlock(nn.Module):

    def __init__(self, config, layer_id, n_layer, use_checkpoint=True):
        super().__init__()

        self.config = config
        self.use_checkpoint = use_checkpoint # Mặc định sử dụng gradient checkpoint

        # Khởi tạo rwkv_args namespace cho giống code gốc nhất
        rwkv_args = types.SimpleNamespace()
        rwkv_args.n_embd = config.hidden_size
        rwkv_args.n_layer = n_layer

        self.ln1 = nn.LayerNorm(rwkv_args.n_embd)
        self.ln2 = nn.LayerNorm(rwkv_args.n_embd)
        self.att = RWKV_TimeMix(rwkv_args, layer_id) # TimeMix được gọi là Attention (att)
        self.ffn = RWKV_ChannelMix(rwkv_args, layer_id) # ChannelMix được gọi là Feedforward (ffn)


    def block_forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def forward(self, x):
        if self.training and self.use_checkpoint:
            x.requires_grad_(True)
            return checkpoint(self.block_forward, x)
        else:
            return self.block_forward(x)


    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        assert layer_index is not None
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval().half()
        try:
            weight_filename = os.path.join(model_path, f"pytorch_{layer_index}.pt")
            module.load_state_dict( torch.load(weight_filename) )
        except:
            print(f"Cannot load GPTBlock from <model_path> {model_path}. The module is initialized automatically.")
            module = init_weight(module)
        return module



class GPTEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = nn.LayerNorm(self.embed_dim)


    def forward(self, input_ids, *args, **kargs):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.word_embeddings(input_ids)
        hidden_states = self.word_embeddings_layernorm(hidden_states)
        return hidden_states


    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)

        module = cls(config).eval()
        try:
            weight_filename = os.path.os.path.join(model_path, "pytorch_embs.pt")
            module.load_state_dict( torch.load(weight_filename) )
        except:
            print(f"Cannot load GPTEmbeddings from <model_path> {model_path}. The module is initialized automatically.")
            module = init_weight(module)
        return module



class GPTLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def forward(self, x, *args, **kargs):
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x


    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            weight_filename = os.path.os.path.join(model_path, "pytorch_lm_head.pt")
            module.load_state_dict( torch.load(weight_filename) )
        except:
            print(f"Cannot load GPTLMHead from <model_name> {model_path}. The module is initialized automatically.")
            module = init_weight(module)
        return module

