# script.py
import time
import torch
import torch.nn as nn

from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel as FSDP
from tqdm import tqdm
from xformers.ops import memory_efficient_attention
import xformers.ops as xops
import torch.distributed

FEAT_SIZE = 1024 * 4
MAX_LEN = 256
BATCH_SIZE = 2

class Layer(nn.Module):
    
    def __init__(self, n_feat):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(n_feat, 64, dim_feedforward=4 * n_feat),
            num_layers=16,
        )

    @property
    def device(self):
      return next(self.parameters()).device

    def convert_to_graph(self):
        sample_in = torch.randn(BATCH_SIZE, MAX_LEN, FEAT_SIZE).to(self.device)
        self.transformer = torch.cuda.make_graphed_callables(self.transformer, (sample_in,), num_warmup_iters=11)

    def forward(self, x):
        return self.transformer(x)

###
dtype = torch.bfloat16  # Setting this to torch.float32 makes this code work.
###

torch.distributed.init_process_group(backend="nccl")
device = torch.device(f"cuda:{torch.distributed.get_rank()}")
torch.cuda.set_device(device)

batch = torch.zeros(BATCH_SIZE, MAX_LEN, FEAT_SIZE).to(device)
mha = Layer(FEAT_SIZE).to(device)
mha.convert_to_graph()

mp_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
mha_fsdp = FSDP(mha, use_orig_params=True, device_id=device, mixed_precision=mp_policy)

times = []
pbar = tqdm(range(1_000), desc="Time", total=1_000)
for _ in range(1_000):
    x = torch.randn(BATCH_SIZE, MAX_LEN, FEAT_SIZE).to(device)
    start = time.time()
    y = mha_fsdp(batch)
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)
    if torch.distributed.get_rank() == 0:
        pbar.update(1)

print(f"Average time: {sum(times) / len(times)} ms")