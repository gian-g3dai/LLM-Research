import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import colorsys
import numpy as np
import pandas as pd
from unakin.llm.tokenizers import LlamaTokenizer
from termcolor2 import colored
from matplotlib import pyplot as plt
from colorama import Fore, Back, Style, init


def print_colored_tokens(values, tokens):
    
    # log scale values
    values = np.log(values)
    max_value = max(values)

    for value, token in zip(values, tokens):
        # if value == max_value:
        #     print(f"\033[97;44m{token}\033[0m", end=' ')  # Print in white text on dark blue background for largest value
        # else:
        # Calculate a gradient from light blue to white for smaller values
        gradient_intensity = int(255 * (value / max_value))
        color_code = f"\033[97;48;2;{100 + gradient_intensity};{100 + gradient_intensity};255m"  # White text on light blue to white gradient background
        reset_code = "\033[0m"
        print(f"{color_code}{token}{reset_code}", end=' ')

def main():
  path = "./token_entr/batch_losses/4.pkl"
  tokenizer_path = "/home/fsuser/fsx/codellama/CodeLlama-13b-Instruct/tokenizer.model"
  tokenizer = LlamaTokenizer(model_path=tokenizer_path)

  per_sample_df = pd.read_pickle(path)

  some_sample = per_sample_df.iloc[3]
  # decode token ids
  dec_tokens = [tokenizer.decode([t]) for t in some_sample["token_ids"]]

  print_colored_tokens(some_sample["losses"], dec_tokens)
    

if __name__ == "__main__":
  main()
  

