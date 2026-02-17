import os
from tqdm import tqdm
from itertools import islice
from multiprocessing import Process

from torch.utils.data import DataLoader

from tests.helpers import get_resource_path
from unakin.llm.data.utils import LMTrainingBatch
from unakin.llm.data.code_dataset import (GPTCodeDataset)
from unakin.llm.data.dataset_factory import DatasetFactory
from unakin.llm.data.datasources import (CSharpFilesDatasource,
                                         MetapromptCodeDatasource,
                                         PromptCodeDatasource,
                                         InstructCodeDatasource,
                                          InstructDatasource)
from unakin.llm.data.prompts import LlamaTemplate, InstructGenTemplate
from unakin.llm.tokenizers import LlamaTokenizer

from utils import prompt_response_print


def test_encode_decode():
  MAX_SEQ_LEN = 4096

  tokenizer = LlamaTokenizer(model_path=get_resource_path() + "/tokenizer.model")
  template = LlamaTemplate()
  template.kwargs = {
    "sys": "DEFAULT SYSTEM PROMPT"
  }

  # initialize the dataset
  dataset = GPTCodeDataset(
    datasources=[
      CSharpFilesDatasource(
        root='/home/fsuser/datasets/games_scraped',
        tokenizer=tokenizer,
        prompt_template=template
      ),
      CSharpFilesDatasource(
        root='/home/fsuser/datasets/games',
        tokenizer=tokenizer,
        prompt_template=template
      ),
    ],
    infinite=False,
    tokenizer=tokenizer,
    max_length=MAX_SEQ_LEN,
    recycle_prompt_template=template,
    max_tokens_as_prompt=100,
    recycle_prompt="Here is a piece of code or text. Please continue it with your own code or text."
  )

  offender_bom_char = chr(int("FEFF", 16))

  for sample in islice(dataset, 100):
    input_str, output_str = sample.decode(dataset.tokenizer)

    assert offender_bom_char not in input_str
    assert offender_bom_char not in output_str

    assert isinstance(input_str, str)
    assert isinstance(output_str, str)
    assert len(input_str) > 0
    assert len(output_str) > 0

    assert len(sample.input_tokens) == MAX_SEQ_LEN
    assert len(sample.output_tokens) == MAX_SEQ_LEN
    assert len(sample.training_mask) == MAX_SEQ_LEN
    assert len(sample.positional_ids) == MAX_SEQ_LEN

def test_json_factory():
  tokenizer = LlamaTokenizer(model_path=get_resource_path() + "/tokenizer.model")

  current_file_location = os.path.dirname(os.path.realpath(__file__))

  # initialize the dataset
  dataset = DatasetFactory.gpt_dataset_from_json(tokenizer=tokenizer, 
                                                  config_file_path=os.path.join(
                                                    current_file_location,
                                                    "test_dset_config.json"
                                                  ))

  for sample in islice(dataset, 10):
    input_str, output_str = sample.decode(dataset.tokenizer)

    assert isinstance(input_str, str)
    assert isinstance(output_str, str)
    assert len(input_str) > 0
    assert len(output_str) > 0

    # test same system prompt as in json config
    assert input_str.startswith("<<SYS>>\nDEFAULT SYSTEM PROMPT\n<</SYS>>\n\n[INST]")


def test_json_factory_no_prompt():
  
  tokenizer = LlamaTokenizer(model_path=get_resource_path() + "/tokenizer.model")

  current_file_location = os.path.dirname(os.path.realpath(__file__))

  # initialize the dataset
  dataset = DatasetFactory.gpt_dataset_from_json(tokenizer=tokenizer, 
                                                  config_file_path=os.path.join(
                                                    current_file_location,
                                                    "test_dset_config_no_prompt.json"
                                                  ))

  for sample in islice(dataset, 10):
    input_str, output_str = sample.decode(dataset.tokenizer)

    assert isinstance(input_str, str)
    assert isinstance(output_str, str)
    assert len(input_str) > 0
    assert len(output_str) > 0

    # test same system prompt as in json config
    assert not input_str.startswith("<<SYS>>\nDEFAULT SYSTEM PROMPT")


def test_dataloader_sharding():
  BSZ = 1
  N_WORKERS = 4
  MAX_SEQ_LEN = 4096 * 8

  tokenizer = LlamaTokenizer(model_path=get_resource_path() + "/tokenizer.model")
  template = LlamaTemplate()
  template.kwargs = {
    "sys": "DEFAULT SYSTEM PROMPT"
  }

  # initialize the dataset
  dataset = GPTCodeDataset(
    datasources=[
      CSharpFilesDatasource(
        root='/home/fsuser/datasets/games_scraped',
        tokenizer=tokenizer,
        prompt_template=template
      ),
      CSharpFilesDatasource(
        root='/home/fsuser/datasets/games',
        tokenizer=tokenizer,
        prompt_template=template
      ),
    ],
    tokenizer=tokenizer,
    max_length=MAX_SEQ_LEN,
    recycle_prompt_template=template,
    max_tokens_as_prompt=100,
    recycle_prompt="Here is a piece of code or text. Please continue it with your own code or text."
  )

  dataloader = DataLoader(dataset, batch_size=BSZ, num_workers=N_WORKERS, collate_fn=collate_fn)

  for idx, s in enumerate(dataloader):
    s: LMTrainingBatch    
    if idx == 10:
      break

    # check inputs
    assert len(s.input_tokens) == BSZ
    assert s.input_tokens.shape == (BSZ, MAX_SEQ_LEN)
    # check outputs
    assert len(s.output_tokens) == BSZ
    assert s.output_tokens.shape == (BSZ, MAX_SEQ_LEN)
    #
    assert len(s.training_mask) == BSZ
    assert s.training_mask.shape == (BSZ, MAX_SEQ_LEN)
    # check sequence lengths
    assert len(s.sequence_lengths) == BSZ
    for idx, seq_len in enumerate(s.sequence_lengths):
      assert sum(seq_len) == MAX_SEQ_LEN


def test_dataloader_sharding_world_size_2():
  # TODO need to find a way to return the datasource indices from each worker
  #  and test them all for uniqueness
  BSZ = 1
  N_WORKERS = 8
  MAX_SEQ_LEN = 4096 * 8
  WORLD_SIZE = 2

  tokenizer = LlamaTokenizer(model_path=get_resource_path() + "/tokenizer.model")
  template = LlamaTemplate()
  template.kwargs = {
    "sys": "DEFAULT SYSTEM PROMPT"
  }

  def spawn_workers_and_iterate(process_rank: int):
    dataset = GPTCodeDataset(
      datasources=[
        CSharpFilesDatasource(
          root='/home/fsuser/datasets/games_scraped',
          tokenizer=tokenizer,
          prompt_template=template,
          reformat_code=False
        ),
        CSharpFilesDatasource(
          root='/home/fsuser/datasets/games',
          tokenizer=tokenizer,
          prompt_template=template,
          reformat_code=False
        ),
      ],
      tokenizer=tokenizer,
      max_length=MAX_SEQ_LEN,
      recycle_prompt_template=template,
      process_rank=process_rank,
      world_size=WORLD_SIZE,
      max_tokens_as_prompt=100,
      recycle_prompt="Here is a piece of code or text. Please continue it with your own code or text."
    )
    
    dataloader = DataLoader(dataset, batch_size=BSZ, num_workers=N_WORKERS, collate_fn=collate_fn)

    for idx, s in enumerate(dataloader):
      s: LMTrainingBatch    
      if idx == 10:
        break

      # check inputs
      assert len(s.input_tokens) == BSZ
      assert s.input_tokens.shape == (BSZ, MAX_SEQ_LEN)
      # check outputs
      assert len(s.output_tokens) == BSZ
      assert s.output_tokens.shape == (BSZ, MAX_SEQ_LEN)
      #
      assert len(s.training_mask) == BSZ
      assert s.training_mask.shape == (BSZ, MAX_SEQ_LEN)
      # check sequence lengths
      assert len(s.sequence_lengths) == BSZ
      for idx, seq_len in enumerate(s.sequence_lengths):
        assert sum(seq_len) == MAX_SEQ_LEN

  processes = []

  for process_rank in range(WORLD_SIZE):
    # initialize the dataset
    processes.append(Process(target=spawn_workers_and_iterate, args=(process_rank,)))
  
  for p in processes:
    p.start()

  for p in processes:
    p.join()

def test_dataloader_sharding_world_size_4_no_prompt():
  # TODO need to find a way to return the datasource indices from each worker
  #  and test them all for uniqueness
  BSZ = 4
  N_WORKERS = 8
  MAX_SEQ_LEN = 1024
  WORLD_SIZE = 4

  def spawn_workers_and_iterate(process_rank: int):
    tokenizer = LlamaTokenizer(model_path=get_resource_path() + "/tokenizer.model")

    current_file_location = os.path.dirname(os.path.realpath(__file__))

    # initialize the dataset
    dataset = DatasetFactory.gpt_dataset_from_json(tokenizer=tokenizer, 
                                                    config_file_path=os.path.join(
                                                      current_file_location,
                                                      "test_dset_config_no_prompt.json"
                                                    ),
                                                    process_rank=process_rank,
                                                    world_size=WORLD_SIZE)
    
    dataloader = DataLoader(dataset, batch_size=BSZ, num_workers=N_WORKERS, collate_fn=collate_fn)

    for idx, s in enumerate(dataloader):
      s: LMTrainingBatch    
      # if idx == 10:
      #   break

      # check inputs
      assert s.input_tokens.shape[1] == MAX_SEQ_LEN
      # check outputs
      assert s.output_tokens.shape[1] == MAX_SEQ_LEN
      #
      assert s.training_mask.shape[1] == MAX_SEQ_LEN
      # check sequence lengths
      for idx, seq_len in enumerate(s.sequence_lengths):
        assert sum(seq_len) == MAX_SEQ_LEN

  processes = []

  for process_rank in range(WORLD_SIZE):
    # initialize the dataset
    processes.append(Process(target=spawn_workers_and_iterate, args=(process_rank,)))
  
  for p in processes:
    p.start()

  for p in processes:
    p.join()


def test_dataloader_bsz4_no_prompt():
  BSZ = 4
  N_WORKERS = 4
  MAX_SEQ_LEN = 1024

  tokenizer = LlamaTokenizer(model_path=get_resource_path() + "/tokenizer.model")

  current_file_location = os.path.dirname(os.path.realpath(__file__))

  # initialize the dataset
  dataset = DatasetFactory.gpt_dataset_from_json(tokenizer=tokenizer, 
                                                  config_file_path=os.path.join(
                                                    current_file_location,
                                                    "test_dset_config_no_prompt.json"
                                                  ))

  dataloader = DataLoader(dataset, batch_size=BSZ, num_workers=N_WORKERS, collate_fn=collate_fn)

  for idx, s in enumerate(dataloader):
    s: LMTrainingBatch    
    if idx == 10:
      break

    # check inputs
    assert s.input_tokens.shape[1] == MAX_SEQ_LEN
    # check outputs
    assert s.output_tokens.shape[1] == MAX_SEQ_LEN
    #
    assert s.training_mask.shape[1] == MAX_SEQ_LEN
    # check sequence lengths
    for idx, seq_len in enumerate(s.sequence_lengths):
      assert sum(seq_len) == MAX_SEQ_LEN


def test_dataloader_sharding_bsz4():
  BSZ = 4
  N_WORKERS = 4
  MAX_SEQ_LEN = 4096 * 8

  tokenizer = LlamaTokenizer(model_path=get_resource_path() + "/tokenizer.model")
  template = LlamaTemplate()
  template.kwargs = {
    "sys": "DEFAULT SYSTEM PROMPT"
  }

  # initialize the dataset
  dataset = GPTCodeDataset(
    datasources=[
      CSharpFilesDatasource(
        root='/home/fsuser/datasets/games_scraped',
        tokenizer=tokenizer,
        prompt_template=template
      ),
      CSharpFilesDatasource(
        root='/home/fsuser/datasets/games',
        tokenizer=tokenizer,
        prompt_template=template
      ),
    ],
    tokenizer=tokenizer,
    max_length=MAX_SEQ_LEN,
    recycle_prompt_template=template,
    max_tokens_as_prompt=100,
    recycle_prompt="Here is a piece of code or text. Please continue it with your own code or text."
  )

  dataloader = DataLoader(dataset, batch_size=BSZ, num_workers=N_WORKERS, collate_fn=collate_fn)

  for idx, s in enumerate(dataloader):
    s: LMTrainingBatch    
    if idx == 10:
      break

    # check inputs
    assert len(s.input_tokens) == BSZ
    assert s.input_tokens.shape == (BSZ, MAX_SEQ_LEN)
    # check outputs
    assert len(s.output_tokens) == BSZ
    assert s.output_tokens.shape == (BSZ, MAX_SEQ_LEN)
    #
    assert len(s.training_mask) == BSZ
    assert s.training_mask.shape == (BSZ, MAX_SEQ_LEN)
    # check sequence lengths
    assert len(s.sequence_lengths) == BSZ
    for idx, seq_len in enumerate(s.sequence_lengths):
      assert sum(seq_len) == MAX_SEQ_LEN


def test_encode_decode_prompt_code_ds():
  MAX_SEQ_LEN = 512

  tokenizer = LlamaTokenizer(model_path="/home/ted/fsx/codellama/CodeLlama-13b-Instruct/tokenizer.model")
  template = LlamaTemplate()
  template.kwargs = {
    "sys": "DEFAULT SYSTEM PROMPT"
  }

  # initialize the dataset
  dataset = GPTCodeDataset(
    datasources=[
      PromptCodeDatasource(
        root='/home/ted/datasets/no_game',
        tokenizer=tokenizer,
        prompt_template=template
      ),
    ],
    tokenizer=tokenizer,
    max_length=MAX_SEQ_LEN,
    multisample=False
  )

  for sample in islice(dataset, 100):
    input_str, output_str = sample.decode(dataset.tokenizer)


    assert isinstance(input_str, str)
    assert isinstance(output_str, str)
    assert len(input_str) > 0
    assert len(output_str) > 0

    assert len(sample.input_tokens) == MAX_SEQ_LEN
    assert len(sample.output_tokens) == MAX_SEQ_LEN
    assert len(sample.training_mask) == MAX_SEQ_LEN
    assert len(sample.positional_ids) == MAX_SEQ_LEN


def test_encode_decode_metaprompt_code_ds():
  MAX_SEQ_LEN = 512

  tokenizer = LlamaTokenizer(model_path=get_resource_path() + "/tokenizer.model")
  template = LlamaTemplate()
  template.kwargs = {
    "sys": "DEFAULT SYSTEM PROMPT"
  }

  # initialize the dataset
  dataset = GPTCodeDataset(
    datasources=[
      MetapromptCodeDatasource(
        root='/home/fsuser/datasets/metaprompts',
        tokenizer=tokenizer,
        prompt_template=template
      )
    ],
    tokenizer=tokenizer,
    max_length=MAX_SEQ_LEN,
    multisample=False
  )

  for sample in islice(dataset, 100):
    input_str, output_str = sample.decode(dataset.tokenizer)

    assert isinstance(input_str, str)
    assert isinstance(output_str, str)
    assert len(input_str) > 0
    assert len(output_str) > 0

    assert len(sample.input_tokens) == MAX_SEQ_LEN
    assert len(sample.output_tokens) == MAX_SEQ_LEN
    assert len(sample.training_mask) == MAX_SEQ_LEN
    assert len(sample.positional_ids) == MAX_SEQ_LEN


def test_instruct_code_ds():

  tokenizer = LlamaTokenizer(model_path="/home/ted/fsx/codellama/CodeLlama-13b-Instruct/tokenizer.model")

  # initialize the dataset
  dataset = DatasetFactory.gpt_dataset_from_json(tokenizer=tokenizer, 
    config_file_path="/home/ted/workdir/ncs/ml-research/recipes/configs/datasets/seed_dataset.json",
    process_rank=0,
    world_size=1)

  train_dl = DataLoader(
    dataset, batch_size=1, shuffle=False,
    num_workers=1, collate_fn=dataset.get_collate_fn(),
    pin_memory=True,
  )

  print(len(train_dl))

  seqs_cnt = 0
  for idx, sample in enumerate(train_dl):
    print(sample.sequence_lengths)
    seqs_cnt += len(sample.sequence_lengths[0])
    print(seqs_cnt)


def test_oasst_instruct_ds():

  tokenizer = LlamaTokenizer(model_path="/home/ted/fsx/codellama/CodeLlama-13b-Instruct/tokenizer.model")

  # initialize the dataset
  dataset = DatasetFactory.gpt_dataset_from_json(tokenizer=tokenizer, 
    config_file_path="/home/ted/workdir/ncs/ml-research/recipes/configs/datasets/instruct.json",
    process_rank=0,
    world_size=1)

  train_dl = DataLoader(
    dataset, batch_size=1, shuffle=False,
    num_workers=1, collate_fn=dataset.get_collate_fn(),
    pin_memory=True,
  )

  print(len(train_dl))

  seqs_cnt = 0
  for idx, sample in enumerate(train_dl):
    print(sample.sequence_lengths)

    if idx == 10:
      break

def test_code_file_ds():
  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model")

  # initialize the dataset
  dataset = DatasetFactory.gpt_dataset_from_json(tokenizer=tokenizer, 
    config_file_path="/home/fsuser/workdir/ncs/ml-research/recipes/configs/datasets/comment.json",
    process_rank=0,
    world_size=1)
  
  train_dl = DataLoader(
    dataset, batch_size=1, shuffle=False,
    num_workers=1, collate_fn=dataset.get_collate_fn(),
    pin_memory=True,
  )

  print(len(train_dl))
  print(train_dl.dataset)
  print(train_dl.dataset.datasources[0].metric_list)


  for idx, sample in enumerate(train_dl):
    print(sample.sequence_lengths)

    input_str, output_str = sample.decode(dataset.tokenizer)

    print(input_str)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(output_str)



    if idx == 10:
      break

def test_multi_oasst_ds():
  
    tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model")
  
    # initialize the dataset
    dataset = DatasetFactory.gpt_dataset_from_json(tokenizer=tokenizer, 
      config_file_path="/home/fsuser/workdir/ncs/ml-research/recipes/configs/datasets/multi_turn.json",
      process_rank=0,
      world_size=1)
  
    train_dl = DataLoader(
      dataset, batch_size=1, shuffle=False,
      num_workers=1, collate_fn=dataset.get_collate_fn(),
      pin_memory=True,
    )
  
    print(len(train_dl))
  
    for idx, batch in enumerate(train_dl):

      input_str, output_str = batch.decode(dataset.tokenizer)

      print(input_str)

      break

def test_commit_ds():

  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model")
  
  # initialize the dataset
  dataset = DatasetFactory.gpt_dataset_from_json(tokenizer=tokenizer, 
    config_file_path="/home/fsuser/workdir/ncs/ml-research/recipes/configs/datasets/commit_dataset.json",
    process_rank=0,
    world_size=1)

  train_dl = DataLoader(
    dataset, batch_size=1, shuffle=False,
    num_workers=1, collate_fn=dataset.get_collate_fn(),
    pin_memory=True,
  )

  print(len(train_dl))

  for idx, batch in enumerate(train_dl):

    input_str, output_str = batch.decode(dataset.tokenizer)

    print(input_str[0])

    break

def test_platy_ds():

  tokenizer = LlamaTokenizer(model_path="/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model")
  
  # initialize the dataset
  dataset = DatasetFactory.gpt_dataset_from_json(tokenizer=tokenizer, 
    config_file_path="/home/fsuser/workdir/ncs/ml-research/recipes/configs/datasets/platy_dataset.json",
    process_rank=0,
    world_size=1)

  train_dl = DataLoader(
    dataset, batch_size=1, shuffle=False,
    num_workers=1, collate_fn=dataset.get_collate_fn(),
    pin_memory=True,
  )

  for idx, batch in enumerate(train_dl):

    input_str, output_str = batch.decode(dataset.tokenizer)


def main():
  test_platy_ds()

if __name__ == "__main__":
  main()