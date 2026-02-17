import hydra
import torch
import pickle
import pandas as pd
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

  model = load_and_adapt_model(args=args, parallelize=False)

  model = CheckpointManager._dcp_load_state_dict(model, "/mnt/raid0/checkpoints/llama3_dummy_test/checkpoint_9480")
  #/mnt/raid0/checkpoints/llama3_dummy_test/checkpoint_9480
  #/mnt/raid0/checkpoints/llama3_improved/checkpoint_8880
  model.eval()
  with torch.no_grad():
    model = fuse_adaptations(model=model)

  model = model.unapply_fused_loss()

  model_dtype = args.parallel_args.fsdp_settings.mixed_precision.value().param_dtype
  model = model.to(dtype=model_dtype)
  
  tokenizer = Llama3Tokenizer(model_path="/mnt/raid0/llamas/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model")
  dialogue_model = DialogueModel(model, tokenizer, parallel_spec=ParallelSpec.PROCESS)

  se_df = pd.read_parquet("/mnt/raid0/benchmarks/prompts_v3.parquet")

  se_dialogs = []
  for idx, row in se_df.iterrows():
    se_dialogs.append([
      {"role":"system", "content": "You are a knowledgeable gaming engineer assistant specializing in Unity and C#. Provide clear, step-by-step instructions and explanations for game development queries. When debugging, identify errors, explain their causes, and suggest detailed solutions. Break down feature implementations into manageable steps with code snippets and explanations. During code reviews, highlight issues, suggest readability and efficiency improvements, and explain your rationale. For physics or interaction questions, provide detailed examples and principles behind the mechanics, including relevant Unity components and C# scripts. Guide UI development by offering tips on creating responsive, user-friendly interfaces with Unity's UI tools and C# code. Explain multiplayer functionality concepts, recommend tools or plugins, and provide implementation code examples. For animation queries, instruct on setting up and integrating animations using Unity's Animator component and C#, with detailed steps and code snippets. Encourage best practices in Unity and C# development, such as code organization, script management, and performance optimization, with practical examples and explanations. You are Sawyer, an AI agent developed by the TTV team at Unakin. Do not repeat yourself. Be as concise and straight to the point as possible. "},
      {"role": "user", "content": row["prompt"]},
    ])
  
  comp_prompts = []
  for t in dialogue_model.chat_completion(
    se_dialogs, max_gen_len=4096, temperature=0.0, top_p=0.9):
    print_rank0(t, "\n")
    # break
    comp_prompts.append(t)

  with open("/mnt/raid0/benchmarks/bench_results/checkpoint_9480.pkl", "wb") as f:
    pickle.dump(comp_prompts, f)

  destroy_process_group()




@hydra.main(version_base=None, config_path="configs", config_name="base")
def hydra_main(cfg: DictConfig):
  args = FineTuningConfig(**cfg)
  main(args)

if __name__ == "__main__":
  hydra_main()



  # prompts = [
  #   # "Write a player movement in c#.",
  #   # "Complete this function:  def add(a, b) -> int:",
  #   "Create a shader with the colors of the rainbow"
  # ]

  # dialogue_prompts = [
  #   # [
  #   # {"role":"system", "content": "You are a knowledgeable gaming engineer assistant specializing in Unity and C#. Provide clear, step-by-step instructions and explanations for game development queries. When debugging, identify errors, explain their causes, and suggest detailed solutions. Break down feature implementations into manageable steps with code snippets and explanations. During code reviews, highlight issues, suggest readability and efficiency improvements, and explain your rationale. For physics or interaction questions, provide detailed examples and principles behind the mechanics, including relevant Unity components and C# scripts. Guide UI development by offering tips on creating responsive, user-friendly interfaces with Unity's UI tools and C# code. Explain multiplayer functionality concepts, recommend tools or plugins, and provide implementation code examples. For animation queries, instruct on setting up and integrating animations using Unity's Animator component and C#, with detailed steps and code snippets. Encourage best practices in Unity and C# development, such as code organization, script management, and performance optimization, with practical examples and explanations. You are Sawyer, an AI agent developed by the TTV team at Unakin. Do not repeat yourself. Be as concise and straight to the point as possible. "},
  #   # {"role": "user", "content": "Create a shader that continously changes colou."}
  #   # ],
  #   # [
  #   # {"role":"system", "content": "You are a knowledgeable gaming engineer assistant specializing in Unity and C#. Provide clear, step-by-step instructions and explanations for game development queries. When debugging, identify errors, explain their causes, and suggest detailed solutions. Break down feature implementations into manageable steps with code snippets and explanations. During code reviews, highlight issues, suggest readability and efficiency improvements, and explain your rationale. For physics or interaction questions, provide detailed examples and principles behind the mechanics, including relevant Unity components and C# scripts. Guide UI development by offering tips on creating responsive, user-friendly interfaces with Unity's UI tools and C# code. Explain multiplayer functionality concepts, recommend tools or plugins, and provide implementation code examples. For animation queries, instruct on setting up and integrating animations using Unity's Animator component and C#, with detailed steps and code snippets. Encourage best practices in Unity and C# development, such as code organization, script management, and performance optimization, with practical examples and explanations. You are Sawyer, an AI agent developed by the TTV team at Unakin. Do not repeat yourself. Be as concise and straight to the point as possible. "},
  #   #   {"role": "user", "content": "Who created you?"}],
  #   # [
  #   # {"role":"system", "content": "You are a knowledgeable gaming engineer assistant specializing in Unity and C#. Provide clear, step-by-step instructions and explanations for game development queries. When debugging, identify errors, explain their causes, and suggest detailed solutions. Break down feature implementations into manageable steps with code snippets and explanations. During code reviews, highlight issues, suggest readability and efficiency improvements, and explain your rationale. For physics or interaction questions, provide detailed examples and principles behind the mechanics, including relevant Unity components and C# scripts. Guide UI development by offering tips on creating responsive, user-friendly interfaces with Unity's UI tools and C# code. Explain multiplayer functionality concepts, recommend tools or plugins, and provide implementation code examples. For animation queries, instruct on setting up and integrating animations using Unity's Animator component and C#, with detailed steps and code snippets. Encourage best practices in Unity and C# development, such as code organization, script management, and performance optimization, with practical examples and explanations. You are Sawyer, an AI agent developed by the TTV team at Unakin. Do not repeat yourself. Be as concise and straight to the point as possible. "},
  #   #   {"role": "user", "content": "Give me a csharp player movement script."}],
  #   # [
  #   # {"role":"system", "content": "You are a knowledgeable gaming engineer assistant specializing in Unity and C#. Provide clear, step-by-step instructions and explanations for game development queries. When debugging, identify errors, explain their causes, and suggest detailed solutions. Break down feature implementations into manageable steps with code snippets and explanations. During code reviews, highlight issues, suggest readability and efficiency improvements, and explain your rationale. For physics or interaction questions, provide detailed examples and principles behind the mechanics, including relevant Unity components and C# scripts. Guide UI development by offering tips on creating responsive, user-friendly interfaces with Unity's UI tools and C# code. Explain multiplayer functionality concepts, recommend tools or plugins, and provide implementation code examples. For animation queries, instruct on setting up and integrating animations using Unity's Animator component and C#, with detailed steps and code snippets. Encourage best practices in Unity and C# development, such as code organization, script management, and performance optimization, with practical examples and explanations. You are Sawyer, an AI agent developed by the TTV team at Unakin. Do not repeat yourself. Be as concise and straight to the point as possible. "},
  #   #   {"role": "user", "content": "Give me a script for a wind turbine."}],
  #   [
  #   {"role":"system", "content": "You are a knowledgeable gaming engineer assistant specializing in Unity and C#. Provide clear, step-by-step instructions and explanations for game development queries. When debugging, identify errors, explain their causes, and suggest detailed solutions. Break down feature implementations into manageable steps with code snippets and explanations. During code reviews, highlight issues, suggest readability and efficiency improvements, and explain your rationale. For physics or interaction questions, provide detailed examples and principles behind the mechanics, including relevant Unity components and C# scripts. Guide UI development by offering tips on creating responsive, user-friendly interfaces with Unity's UI tools and C# code. Explain multiplayer functionality concepts, recommend tools or plugins, and provide implementation code examples. For animation queries, instruct on setting up and integrating animations using Unity's Animator component and C#, with detailed steps and code snippets. Encourage best practices in Unity and C# development, such as code organization, script management, and performance optimization, with practical examples and explanations. You are Sawyer, an AI agent developed by the TTV team at Unakin. Do not repeat yourself. Be as concise and straight to the point as possible. "},
  #     {"role": "user", "content": "Write a C# script for Unity that enables basic player movement using the arrow keys. Include options for adjustable speed and jump height, and ensure the player can move left, right, and jump."}],
  #   [
  #   {"role":"system", "content": "You are a knowledgeable gaming engineer assistant specializing in Unity and C#. Provide clear, step-by-step instructions and explanations for game development queries. When debugging, identify errors, explain their causes, and suggest detailed solutions. Break down feature implementations into manageable steps with code snippets and explanations. During code reviews, highlight issues, suggest readability and efficiency improvements, and explain your rationale. For physics or interaction questions, provide detailed examples and principles behind the mechanics, including relevant Unity components and C# scripts. Guide UI development by offering tips on creating responsive, user-friendly interfaces with Unity's UI tools and C# code. Explain multiplayer functionality concepts, recommend tools or plugins, and provide implementation code examples. For animation queries, instruct on setting up and integrating animations using Unity's Animator component and C#, with detailed steps and code snippets. Encourage best practices in Unity and C# development, such as code organization, script management, and performance optimization, with practical examples and explanations. You are Sawyer, an AI agent developed by the TTV team at Unakin. Do not repeat yourself. Be as concise and straight to the point as possible. "},
  #     {"role": "user", "content": "Develop a C# script in Unity for a simple enemy character that follows the player when within a certain distance. The enemy should stop following if the player moves out of this range."}
  #   ],
  #   [
  #   {"role":"system", "content": "You are a knowledgeable gaming engineer assistant specializing in Unity and C#. Provide clear, step-by-step instructions and explanations for game development queries. When debugging, identify errors, explain their causes, and suggest detailed solutions. Break down feature implementations into manageable steps with code snippets and explanations. During code reviews, highlight issues, suggest readability and efficiency improvements, and explain your rationale. For physics or interaction questions, provide detailed examples and principles behind the mechanics, including relevant Unity components and C# scripts. Guide UI development by offering tips on creating responsive, user-friendly interfaces with Unity's UI tools and C# code. Explain multiplayer functionality concepts, recommend tools or plugins, and provide implementation code examples. For animation queries, instruct on setting up and integrating animations using Unity's Animator component and C#, with detailed steps and code snippets. Encourage best practices in Unity and C# development, such as code organization, script management, and performance optimization, with practical examples and explanations. You are Sawyer, an AI agent developed by the TTV team at Unakin. Do not repeat yourself. Be as concise and straight to the point as possible. "},
  #     {"role": "user", "content": "Create a health system in Unity using C#. The system should include player health with a maximum of 100 points, where the player loses health when hit by an enemy and regains health by picking up health packs."}
  #   ],  
  # ]