import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tests.helpers import get_resource_path
from utils import prompt_response_print
from unakin.llm.data.datasources import (CSharpFilesDatasource,
                                         MetapromptCodeDatasource,
                                         PromptCodeDatasource,
                                         InstructCodeDatasource,
                                         InstructDatasource,
                                          CommDatasource,
                                          MultiTurnOasstDatasource)
from unakin.llm.data.datasources.code_datasource import PromptTokensSample
from unakin.llm.data.prompts import LlamaTemplate, InstructGenTemplate, MultiTurnTemplate
from unakin.llm.tokenizers import LlamaTokenizer
from unakin.llm.data.dataset_factory import DatasetFactory


def test_open():
    template = LlamaTemplate()
    template.kwargs = {
        "sys": "DEFAULT SYSTEM PROMPT"
    }

    datasource = CSharpFilesDatasource(
        root='/home/fsuser/datasets/games_scraped',
        tokenizer=LlamaTokenizer(model_path=get_resource_path() + "/tokenizer.model"),
        prompt_template=template
    )

    for i in range(min(10, len(datasource))):
        sample = datasource[i]
        assert isinstance(sample, PromptTokensSample)
        assert len(sample.prompt_tokens) > 0
        assert len(sample.sample_tokens) > 0
        assert isinstance(sample.prompt_tokens[0], int)
        assert isinstance(sample.sample_tokens[0], int)

def test_dataloader_sharding_bs1():
    template = LlamaTemplate()
    template.kwargs = {
        "sys": "DEFAULT SYSTEM PROMPT"
    }

    datasource = CSharpFilesDatasource(
        root='/home/fsuser/datasets/games_scraped',
        tokenizer=LlamaTokenizer(model_path=get_resource_path() + "/tokenizer.model"),
        prompt_template=template
    )

    dataloader = DataLoader(datasource, batch_size=1, num_workers=4)
    for idx, sample in enumerate(dataloader):
        if idx == 10:
            break
        assert isinstance(sample, PromptTokensSample)
        assert len(sample.prompt_tokens) > 0
        assert len(sample.sample_tokens) > 0
        assert isinstance(sample.prompt_tokens[0], torch.Tensor)
        assert isinstance(sample.sample_tokens[0], torch.Tensor)

def test_train_test_split():

    datasource_train = CSharpFilesDatasource(
        root='/home/fsuser/datasets/games_scraped',
        tokenizer=LlamaTokenizer(model_path=get_resource_path() + "/tokenizer.model"),
        percent_skip=0,
        percent_use=20,
    )

    datasource_test = CSharpFilesDatasource(
        root='/home/fsuser/datasets/games_scraped',
        tokenizer=LlamaTokenizer(model_path=get_resource_path() + "/tokenizer.model"),
        percent_skip=20,
        percent_use=80,
    )

    for i in range(min(10, len(datasource_train))):
        sample = datasource_train[i]
        assert isinstance(sample, PromptTokensSample)
        assert len(sample.prompt_tokens) == 0
        assert len(sample.sample_tokens) > 0
        assert isinstance(sample.sample_tokens[0], int)

    for i in range(min(10, len(datasource_test))):
        sample = datasource_test[i]
        assert isinstance(sample, PromptTokensSample)
        assert len(sample.prompt_tokens) == 0
        assert len(sample.sample_tokens) > 0
        assert isinstance(sample.sample_tokens[0], int)

def test_prompt_code_datasource():

  template = LlamaTemplate()
  template.kwargs = {
      "sys": "DEFAULT SYSTEM PROMPT"
  }


  tokenizer = LlamaTokenizer(model_path="/home/ted/fsx/codellama/CodeLlama-13b-Instruct/tokenizer.model")

  datasource = PromptCodeDatasource(
      root='/home/ted/datasets/no_game',
      tokenizer=LlamaTokenizer(model_path="/home/ted/fsx/codellama/CodeLlama-13b-Instruct/tokenizer.model"),
      prompt_template=template
  )

  for i in range(min(10, len(datasource))):
    sample = datasource[i]
    print(sample)
    print("PROMPT TOKENS")
    print(tokenizer.decode(sample.prompt_tokens))
    print("SAMPLE TOKENS")
    print(tokenizer.decode(sample.sample_tokens))
    assert isinstance(sample, PromptTokensSample)
    assert len(sample.prompt_tokens) > 0
    assert len(sample.sample_tokens) > 0
    assert isinstance(sample.prompt_tokens[0], int)
    assert isinstance(sample.sample_tokens[0], int)

  exit()

def test_metaprompt_code_datasource():

  template = LlamaTemplate()
  template.kwargs = {
      "sys": "DEFAULT SYSTEM PROMPT"
  }

  tokenizer = LlamaTokenizer(model_path="/home/ted/fsx/codellama/CodeLlama-13b-Instruct/tokenizer.model")

  datasource = MetapromptCodeDatasource(
      root='/home/ted/datasets/metaprompts',
      tokenizer=tokenizer,
      prompt_template=template
  )

  for i in range(min(10, len(datasource))):
    sample = datasource[i]
    print(sample)
    print("PROMPT TOKENS")
    print(tokenizer.decode(sample.prompt_tokens))
    print("SAMPLE TOKENS")
    print(tokenizer.decode(sample.sample_tokens))
    assert isinstance(sample, PromptTokensSample)
    assert len(sample.prompt_tokens) > 0
    assert len(sample.sample_tokens) > 0
    assert isinstance(sample.prompt_tokens[0], int)
    assert isinstance(sample.sample_tokens[0], int)


def test_instruct_code_datasource():
   
  template = InstructGenTemplate(
    base_prompt="""[INST] {instruction}\n{input} [/INST]"""
  )
  template.kwargs = {
        "sys": "DEFAULT SYSTEM PROMPT",
        "instruction": ["Develop a {code_language} program that not only accomplishes the provided task efficiently but also adheres to industry best practices. Focus on writing clean, readable, and maintainable code. Ensure comprehensive commenting and documentation for clear understanding. Optimize algorithms and data structures for performance. Rigorously test the code for accuracy, considering potential edge cases and handling exceptions gracefully. Aim for code that exhibits both elegance and resilience, showcasing proficiency in {code_language}.",
          "Craft a {code_language} program that addresses the assigned task with an emphasis on code quality. Pay close attention to readability, maintainability, and scalability. Implement thorough comments and documentation to facilitate comprehension. Optimize algorithms and data structures for optimal performance. Conduct comprehensive testing, accounting for edge cases and error handling. Strive for code that is not only functional but also elegant and robust, showcasing your proficiency in {code_language}.",
          "In {code_language}, create a program that effectively addresses the provided task. It is crucial that your code is of high quality, adhering to best practices and industry standards. Focus on making it readable, maintainable, and scalable. Ensure that you incorporate comprehensive comments and documentation for better understanding. Optimize algorithms and data structures for efficiency. Rigorously test your code for correctness, considering potential edge cases and handling exceptions gracefully. Aim for code that is both elegant and robust, demonstrating proficiency in {code_language}.",
          "Develop a high-quality {code_language} program to accomplish the given task. Prioritize readability, maintainability, and scalability in your code. Include detailed comments and documentation for clear comprehension. Optimize algorithms and data structures for efficiency. Conduct rigorous testing to ensure accuracy, considering edge cases and error handling. Strive for code that is both elegant and resilient, showcasing your proficiency in {code_language}.",
          "Write a {code_language} program that effectively fulfills the provided task. It is imperative that your code meets high-quality standards, including readability, maintainability, and scalability. Implement comprehensive comments and documentation to enhance understanding. Optimize algorithms and data structures for efficient performance. Thoroughly test your code for correctness, accounting for potential edge cases and exception handling. Aim for code that is both elegant and robust, demonstrating your proficiency in {code_language}.",
          "Compose a {code_language} program that not only achieves the assigned objective but also meets stringent quality standards. Emphasize readability, maintainability, and scalability in your code. Include thorough comments and documentation for clear comprehension. Optimize algorithms and data structures for efficiency. Conduct comprehensive testing to ensure correctness, taking into account edge cases and exception handling. Strive for code that is both elegant and resilient, showcasing your proficiency in {code_language}.",
          "Produce a {code_language} program that effectively addresses the given task while adhering to high-quality coding practices. Prioritize readability, maintainability, and scalability. Include detailed comments and documentation for better comprehension. Optimize algorithms and data structures for efficient performance. Rigorously test your code for accuracy, considering potential edge cases and handling exceptions gracefully. Aim for code that is both elegant and robust, demonstrating your proficiency in {code_language}.",
          "Create a {code_language} program that effectively accomplishes the provided task. Ensure that your code is of high quality, emphasizing readability, maintainability, and scalability. Incorporate comprehensive comments and documentation for better understanding. Optimize algorithms and data structures for efficiency. Rigorously test your code for correctness, taking into account potential edge cases and exception handling. Strive for code that is both elegant and resilient, showcasing your proficiency in {code_language}.",
          "Develop a {code_language} program that addresses the given task with a strong focus on code quality. Pay attention to readability, maintainability, and scalability. Incorporate thorough comments and documentation for clarity. Optimize algorithms and data structures for performance. Conduct comprehensive testing, including consideration of edge cases and error handling. Strive for code that is not only functional but also elegant and robust, highlighting your proficiency in {code_language}.",
          "Craft a {code_language} program that effectively tackles the assigned task. It is crucial that your code meets high standards of quality, including readability, maintainability, and scalability. Incorporate detailed comments and documentation for clear understanding. Optimize algorithms and data structures for efficient performance. Rigorously test your code for correctness, taking into consideration potential edge cases and handling exceptions gracefully. Aim for code that is both elegant and robust, showcasing your proficiency in {code_language}."],
    }   

  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model")

  datasource = InstructCodeDatasource(
        root='/mnt/raid0/datasets/code_contests',
        tokenizer=tokenizer,
        prompt_template=template,
        solutions_per_problem=1500,
        split="test",
        reversed=False,
  )

  for idx, sample in tqdm(enumerate(datasource), total=len(datasource)):
    
    prompt_response_print(sample.prompt_tokens, sample.sample_tokens, tokenizer)

    if idx == 4:
      break

def test_oasst_datasource():
  template = InstructGenTemplate(
    base_prompt="""[INST]\n{instruction}\n\n{input}[/INST]"""
  )
  template.kwargs = {
        "instruction": ["Given the following prompt or question, respond accordingly.", "Some other prompt."],
    }   

  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model")

  datasource = InstructDatasource(
        root='/mnt/raid0/datasets/oasst',
        tokenizer=tokenizer,
        prompt_template=template,
        split="train_fst2",
        reversed=False,
  )


  for idx, sample in tqdm(enumerate(datasource), total=len(datasource)):
    
    prompt_response_print(sample.prompt_tokens, sample.sample_tokens, tokenizer)

    if idx == 4:
      break

def test_comment_code_datasource():
  template = InstructGenTemplate(
    base_prompt="""[INST]\n{instruction}\n\n{input}[/INST]"""
  )
  template.kwargs = {
        "instruction": ["Given the following prompt or question, respond accordingly.", "Some other prompt."],
    }   

  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model")

  datasource = CommDatasource(
        root='/mnt/raid0/datasets/comments',
        tokenizer=tokenizer,
        prompt_template=template,
        split="train",
        # use_triple_backticks=True,
        # normalize_line_endings=True,
  )


  for idx, sample in tqdm(enumerate(datasource), total=len(datasource)):
    
    prompt_response_print(sample.prompt_tokens[:100], sample.sample_tokens[:100], tokenizer)

    if idx == 4:
      break
  

def test_multi_oasst_datasource():
  template_prompt = MultiTurnTemplate(
    base_prompt="""\n [INST] {instruction}\n{turn_text} [/INST] \n"""
  )
  template_prompt.kwargs = {
        "instruction": ["Given the following prompt or question, respond accordingly.", "Some other prompt."],
    }
  
  template_asst = MultiTurnTemplate(
    base_prompt="""[RESP]\n{turn_text} [/RESP] """
  )
     

  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model")

  datasource = MultiTurnOasstDatasource(
        root='/mnt/raid0/datasets/multi_oasst',
        tokenizer=tokenizer,
        prompter_template=template_prompt,
        assistant_template=template_asst,
        split="train",
        percent_skip=0,
        percent_use=100,
        metric_list=[],
  )

  for idx, sample in tqdm(enumerate(datasource), total=len(datasource)):

    input_tokens, output_tokens, mask = sample.input_tokens, sample.output_tokens, sample.training_mask

    # color the sample according to the mask
    colored_input_tokens = []
    
    for i, token in enumerate(input_tokens):
      if mask[i] == 1:
        colored_input_tokens.append("\033[91m" + tokenizer.decode([token]) + "\033[0m")
      else:
        colored_input_tokens.append(tokenizer.decode([token]))

    print("INPUT")
    print(" ".join(colored_input_tokens))

    # print(sample) 

    if idx > 10:
      break





if __name__ == "__main__":
    # test_prompt_code_datasource()
    # test_metaprompt_code_datasource()
    # test_instruct_code_datasource()
    # test_comment_code_datasource()
  test_multi_oasst_datasource()