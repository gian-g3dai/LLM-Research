from unakin.llm import BaseLLM
import torch

if __name__ == "__main__":
  print(BaseLLM.__annotations__)
  embedding_table = torch.nn.Embedding(32_000, 5120)
  print(embedding_table.weight.shape)