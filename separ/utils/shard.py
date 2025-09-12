
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLayer
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.xlnet.modeling_xlnet import XLNetLayer
from torch.distributed.fsdp import fully_shard

from torch import nn 
import os, torch 
import torch.distributed as dist 

WRAP_MODULES = {GPT2Block, XLMRobertaLayer, BertLayer,LlamaDecoderLayer, XLNetLayer}

# def recursive_shard(model: nn.Module):
#     """Applies fully_shard recursively to the wrapping modules."""
#     if any(isinstance(model, m) for m in WRAP_MODULES) and not model.__class__.__name__.startswith('FSDP'):
#         fully_shard(model)
#         return
#     if hasattr(model, 'modules'):
#         for m in list(model.modules())[1:]:
#             recursive_shard(m)
#     elif isinstance(model, nn.ModuleList):
#         for m in model:
#             recursive_shard(m)

def shard_modules(model: nn.Module):
    for module in model.modules():
        if any(isinstance(module, m) for m in WRAP_MODULES):
            fully_shard(module)
            

def is_distributed() -> bool:
    return "LOCAL_RANK" in os.environ or "TORCHELASTIC_RUN_ID" in os.environ

WORLD_SIZE = int(os.environ['WORLD_SIZE']) if is_distributed() else 1


def local_rank() -> int:
    return int(os.environ['LOCAL_RANK'])

def is_main() -> bool:
    return not is_distributed() or local_rank() ==  0

def sync(x: torch.Tensor, rank: int | None = None):
    if rank is None:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    else:
        dist.broadcast(x, rank)
        
        
def all_gather_objects(objects: list) -> list:
    gathered = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(gathered, objects)
    unified = []
    for obj in gathered:
        unified += obj 
    return unified 
    
    
    