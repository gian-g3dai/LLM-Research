# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import fire
import json

from tqdm import tqdm
from llama import Dialog, Llama
from typing import List, Optional

def gen_samples_from_df():
    pass

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    def process_file(filename):
      with open(filename, 'r') as file:
          lines = file.readlines()

      processed_lines = []
      for line in lines:
          # Remove the number at the beginning of each line
          processed_line = line.split('. ', 1)[-1].strip()
          processed_lines.append(processed_line)

      return processed_lines
    
    def get_prompts_from_json(filepath):

        with open(filepath, 'r') as f:
            data = json.load(f)

        prompts = []
        for sample in data:
            prompts.append((sample["prompt"], sample["response"]))

        return prompts
        

    prompt_batch = max_batch_size

    # filename = "benchmark-prompts-v3.txt"
    # processed_lines = process_file(filename)

    filepath = "/mnt/raid0/datasets/hf/platy_4rw.json"

    # get ptompts from file
    processed_lines = get_prompts_from_json(filepath)

    
    ann_data = []

    for i in tqdm(range(0, len(processed_lines), prompt_batch)):
        curr_lines = processed_lines[i:i+prompt_batch]

        dialogs: List[Dialog] = [
            [
                # {
                #     "role": "system",
                #     "content": "You are a coding assistant that specialize in C# game development.",
                # },
                {"role": "user", "content": p},
                {"role": "assistant", "content": r},
                {"role": "user", "content": "Rewrite the previous reponse as a teacher would respond like. Retain the formatting that was used to respond such as LaTeX. Do not reference that the response is rewritten."},
            ] for p, r in curr_lines
        ]

        # dialogs: List[Dialog] = [
        #             [
        #                 # {
        #                 #     "role": "system",
        #                 #     "content": "You are a coding assistant that specialize in C# game development.",
        #                 # },
        #                 {"role": "user", "content": bench_prompt},
        #             ] for bench_prompt in curr_lines
        #         ]

        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for result in results:
            print(
                f"{result['generation']['prompt']}"
            )
            print("\n==================================\n")

            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")

            ann_data.append({
                "prompt": result['generation']['prompt'],
                "response": result['generation']['content'],
            })
        
    # Specify the file path
    file_path = "platy_rewrite.json"

    # Dump the list of dictionaries to the file
    with open(file_path, 'w') as json_file:
        json.dump(ann_data, json_file, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
