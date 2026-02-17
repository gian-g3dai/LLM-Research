

from unakin.llm.tokenizers import LlamaTokenizer
from unakin.llm._data.datasources.datasource_loader import get_datasource

def main():
  config_file_path = "deepm_code_contests.yaml"
  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model")
  data_root = "/mnt/raid0/datasets/"
  
  data = get_datasource(config_file_path, tokenizer, data_root=data_root)

if __name__ == '__main__':
  main()
