import os
import hydra
import torch
import json
import pickle
from unakin.llm.distributed.utils import (
  PyObjShard,
  dist_partition_py_type,
  dist_gather_py_type,
  get_rank,
  print_rank0,
  setup_distributed_process_group,
  destroy_process_group,
  DistributedList,
  ParallelSpec
)
from omegaconf import DictConfig, OmegaConf
from unakin.llm.dialogue.base_new import DialogueModel, Dialog
from unakin.llm.models.llama.model_compile import Transformer, TransformerBlock
from unakin.llm.adapters.llm_adapter_new import fuse_adaptations
from unakin.llm.tokenizers import LlamaTokenizer, Llama3Tokenizer
from unakin.llm.models.llama.model_compile import CODE_LLAMA_13B_WEIGHTS, Transformer, KVCache, LLAMA3_70B_WEIGHTS, LLAMA3_8B_WEIGHTS
from finetuning import CheckpointManager, ExperimentState, FineTuningConfig, load_and_adapt_model


def str_to_prompt(prompt: str) -> Dialog:
  return [{"role": "user", "content": prompt}]

def main(args: FineTuningConfig):
  
  setup_distributed_process_group()


  print(args.base_model_args.base_path)

  model = load_and_adapt_model(args=args, parallelize=False)

  trained_rope_theta = model.theta

  model = CheckpointManager._dcp_load_state_dict(model, "/mnt/raid0/checkpoints/llama3_dummy_test/checkpoint_9480")
  model.eval()
  with torch.no_grad():
    model = fuse_adaptations(model=model)

  model = model.unapply_fused_loss()

  model_dtype = args.parallel_args.fsdp_settings.mixed_precision.value().param_dtype
  model = model.to(dtype=model_dtype)

  og_params_path = os.path.join("/mnt/raid0/llamas/llama3/Meta-Llama-3-8B-Instruct/", "params.json")

  og_params = json.load(open(og_params_path, "r"))
  # mod rope theta to be the one used at training
  og_params["rope_theta"] = trained_rope_theta

  print("ROPE:", trained_rope_theta)

  params_out_path = os.path.join("/mnt/raid0/llamas/llama3/Meta-Llama-3-8B-CKPT", "params.json")
  json.dump(og_params, open(params_out_path, "w"))

  torch.save(model.state_dict(), os.path.join("/mnt/raid0/llamas/llama3/Meta-Llama-3-8B-CKPT", "consolidated.00.pth"))

  destroy_process_group()

@hydra.main(version_base=None, config_path="configs", config_name="base")
def hydra_main(cfg: DictConfig):
  args = FineTuningConfig(**cfg)
  main(args)

if __name__ == "__main__":
  hydra_main()

