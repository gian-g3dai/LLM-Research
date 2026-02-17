from argparse import ArgumentParser

import torch.distributed

from unakin.llm.models.llama.utils import *

if __name__ == "__main__":
  """
  This script converts the LLaMa v2 weight checkpoints into a single checkpoint.

  Sample usage:
  ```
  torchrun --nproc_per_node 4 convert_llama_checkpoint.py \
      --llama_v2_dir=/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct \
      --tokenizer_path=/mnt/raid0/llamas/codellama/CodeLlama-34b-Instruct/tokenizer.model \
      --mono_ckpt_path=/mnt/raid0/llamas/llama_base/code-llama-34b-instruct.pth \

  torchrun --nproc_per_node 4 convert_llama_checkpoint.py \
      --llama_v2_dir=/mnt/raid0/llamas/CodeLlama-34b-Instruct \
      --tokenizer_path=/mnt/raid0/llamas/CodeLlama-34b-Instruct/tokenizer.model \
      --mono_ckpt_path=/mnt/raid0/llama_base/code-llama-34b-instruct.pth \
  ```

  Make sure to launch this script with the appropiate value for `--nproc_per_node`.
  This is because LLaMa v2 build method explicitly requires this number to be equal to
  the model parallelism degree of the selected checkpoint.
  """

  parser = ArgumentParser()
  parser.add_argument("--llama_v2_dir", type=str,
                      help="Path to LLaMa v2 weight checkpoints directory")
  parser.add_argument("--tokenizer_path", type=str,
                      help="Path to tokenizer model")
  parser.add_argument("--mono_ckpt_path", type=str,
                      help="Name with which the consolidated weight checkpoint will be saved")
  args = parser.parse_args()

  model = load_llama_checkpoint(
    args.llama_v2_dir,
    args.tokenizer_path,
    mode_parallel_size=get_model_parallelism_requirement(args.llama_v2_dir),
    # TODO: Not particularly fond of this initialization scheme
    # TODO: but I won't judge Meta and Hugo Touvron for it
    max_seq_len=1,
    max_batch_size=1)

  if torch.distributed.get_rank() == 0:
    torch.save(model.state_dict(), args.mono_ckpt_path)
  """
  convert("/home/fsuser/fsx/llama/llama-2-13b-chat", "/home/fsuser/fsx/llama_base/llama-13b.pth")
  """