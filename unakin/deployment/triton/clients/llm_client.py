from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
import multiprocessing
import threading
import time
from typing import Any, Dict, Generator, List, Optional
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype
import queue
import numpy as np
import tritonclient
import tritonclient.grpc

import os

from unakin.llm.tokenizers.base import TokenizerBase
from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer
from unakin.deployment.triton.clients.llm_repeat_karp import int_list_has_repeats



#TOKENIZER_PATH = "/home/ubuntu/codellama_ckpt/13B_tokenizer/tokenizer.model"
TOKENIZER_PATH = "/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model"

MAX_SIZE = 4096
tokenizer = LlamaTokenizer(model_path=TOKENIZER_PATH)

class RetryLimitExceeded(Exception):
  pass

class Prompt:
  
  def __init__(self, prompt: str):
    self.prompt = prompt
    self._toks = None

  def expand(self, input: str):
    self.prompt = self.prompt + input

  def get_ids(self, tokenizer: TokenizerBase):
    # this implements token healing
    # it's also necessary because of how dumb the request id cacher is 
    if self._toks is None:
      self._toks = tokenizer.tokenize(self.prompt, bos=True, eos=False)[0]
    return self._toks
  
  def get_text(self):
    return self.prompt
  
  def __add__(self, other):
    return Prompt(" ".join([self.prompt, other.prompt]))
  
  @staticmethod
  def from_ids(tokenizer: TokenizerBase, ids: List[int]):
    text = tokenizer.decode(ids)
    return text
  
class StreamChunkAccumulator:
  
  def __init__(self, chunk_prebuffering: int = 64):
    self._queue = queue.Queue()
    self.chunk_prebuffering = chunk_prebuffering
    self._chunks_ready_event = threading.Event()
    self._has_finished = False
    self.size = 0

  def add_chunk(self, chunk: int):
    self.size += 1
    self._queue.put(chunk)
    if self._queue.qsize() >= self.chunk_prebuffering:
      self._chunks_ready_event.set()

  def add_bulk(self, chunks: List[int]):
    for chunk in chunks:
      self.add_chunk(chunk)

  def get(self):
   while True:
    if self._chunks_ready_event.is_set() and self._queue.empty():
      break

    while not self._queue.empty():
      chunk = self._queue.get()
      yield chunk

    time.sleep(0.05)
  
  def has_finished(self):
    self._has_finished = True
    self._chunks_ready_event.set()

  def reset(self):
    self._queue = queue.Queue()
    self._chunks_ready_event.clear()

@dataclass(frozen=False)
class LLMRequest:
  url: str
  model_name: str
  request_id: str
  prompt: Prompt
  tokens: List[int]
  tokenizer: TokenizerBase
  repetition_penalty: float
  presence_penalty: float
  temperature: float
  max_gen_len: int
  stop_words: List[int]
  bad_words: List[int]
  min_repeat_size: int
  max_repeat_size: int
  min_repeat_proportion: float
  repeat_retries: int

class TritonCommsProtocol(str, Enum):
  HTTP = "http"
  GRPC = "grpc"

class UserDataCollector:

  def __init__(self):
    self._completed_requests = queue.Queue()

def callback(user_data: UserDataCollector, result: Any, error: Any):
  if error:
    user_data._completed_requests.put(error)
  else:
    user_data._completed_requests.put(result)
    

def prepare_triton_tensor(name: str, input: np.ndarray, protocol: TritonCommsProtocol) -> tritonclient.grpc.InferInput:
  client_util = httpclient if protocol == TritonCommsProtocol.HTTP else grpcclient
  tensor = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
  tensor.set_data_from_numpy(input)
  return tensor

def prepare_output_tensor(name: List[str], protocol: TritonCommsProtocol) -> List[tritonclient.grpc.InferRequestedOutput]:
  client_util = httpclient if protocol == TritonCommsProtocol.HTTP else grpcclient
  for n in name:
    output = client_util.InferRequestedOutput(name)
  return output

def prepare_trt_triton_prompt(
  system_prompt: Optional[Prompt],
  prompt: Optional[Prompt],
  tokenizer: TokenizerBase,
  log_probs: bool = False,
  input_tokens: Optional[List[int]] = None,
  protocol: TritonCommsProtocol = TritonCommsProtocol.GRPC,
):
  
  if prompt and input_tokens:
    raise ValueError("Cannot specify both `prompt` and `input_tokens`")

  if not input_tokens or len(input_tokens) == 0:
    if system_prompt is not None:
      prompt = system_prompt + prompt
    input_ids = prompt.get_ids(tokenizer)
  else:
    input_ids = input_tokens
  
  input_length = [[len(input_ids)]]
  input_ids_data = np.array([input_ids]).astype(np.int32)
  input_length_data = np.array(input_length, dtype=np.int32)
  end_id = np.array([[tokenizer.eos_token]]).astype(np.uint32)

  inputs = [
    prepare_triton_tensor("input_ids", input_ids_data, protocol),
    prepare_triton_tensor("input_lengths", input_length_data, protocol),
    #prepare_triton_tensor("end_id", end_id, protocol),
  ]

  if log_probs:
    log_probs_data = np.array([[True]]).astype(bool)
    inputs.append(prepare_triton_tensor("return_log_probs", log_probs_data, protocol))

  return inputs

def prepare_trt_triton_params(
  temperature: float = 0.6,
  max_gen_len: int = 512,
  runtime_top_k: Optional[int] = None,
  runtime_top_p: Optional[float] = None,
  presence_penalty: Optional[float] = None,
  repetition_penalty: Optional[float] = None,  
  stop_words: Optional[List[int]] = None,
  bad_words: Optional[List[int]] = None,
  protocol: TritonCommsProtocol = TritonCommsProtocol.GRPC,
) -> List[tritonclient.grpc.InferInput]:
  
  temperature_data = np.array([[temperature]], dtype=np.float32)
  max_gen_len_data = np.array([[max_gen_len]], dtype=np.uint32)

  inputs = [
    prepare_triton_tensor("temperature", temperature_data, protocol),
    prepare_triton_tensor("request_output_len", max_gen_len_data, protocol),
  ]

  if runtime_top_k and runtime_top_p:
    raise ValueError("Cannot specify both `runtime_top_k` and `runtime_top_p`")

  if runtime_top_k:
    runtime_top_k_data = np.array([[runtime_top_k]], dtype=np.uint32)
    inputs.append(prepare_triton_tensor("runtime_top_k", runtime_top_k_data, protocol))

  if runtime_top_p:
    runtime_top_p_data = np.array([[runtime_top_p]], dtype=np.float32)
    inputs.append(prepare_triton_tensor("runtime_top_p", runtime_top_p_data, protocol))

  if presence_penalty and repetition_penalty:
    raise ValueError("Cannot specify both `presence_penalty` and `repetition_penalty`")

  if presence_penalty:
    presence_penalty_data = np.array([[presence_penalty]], dtype=np.float32)
    inputs.append(prepare_triton_tensor("presence_penalty", presence_penalty_data, protocol))

  if repetition_penalty:
    repetition_penalty_data = np.array([[repetition_penalty]], dtype=np.float32)
    inputs.append(prepare_triton_tensor("repetition_penalty", repetition_penalty_data, protocol))

  return inputs

def prepare_trt_triton_stream(
  protocol: TritonCommsProtocol = TritonCommsProtocol.GRPC,
) -> List[tritonclient.grpc.InferInput]:
  
    streaming_data = np.array([[True]], dtype=bool)
    beam_width_data = np.array([[1]]).astype(np.uint32)
    
    inputs = [
      prepare_triton_tensor("streaming", streaming_data, protocol),
      prepare_triton_tensor("beam_width", beam_width_data, protocol),
    ]
  
    return inputs

def prepare_trt_triton_outputs(
  output_ids: bool = True,
  cum_log_probs: bool = True,
) -> List[tritonclient.grpc.InferRequestedOutput]:
    
    outputs = []
  
    if output_ids:
      outputs.append(grpcclient.InferRequestedOutput("output_ids"))
  
    if cum_log_probs:
      outputs.append(grpcclient.InferRequestedOutput("cum_log_probs"))
  
    return outputs

def seq_has_eos(output_ids: List[int], eos_token: int) -> bool:
  for i in range(len(output_ids)):
    if output_ids[i] == eos_token:
      return True
  return False

def trim_to_eos(output_ids: List[int], eos_token: int) -> List[int]:
  for i in range(len(output_ids)):
    if output_ids[i] == eos_token:
      return output_ids[:i]
  return output_ids

def run_stream(
    client: tritonclient.grpc.InferenceServerClient,
    request_flags: LLMRequest,
    trim_output_to_eos: bool = True,
    stream_is_running: bool = False,
    stream_collator: UserDataCollector = None,
    collection_queue: queue.Queue = None,):
  model_name = request_flags.model_name
  tokenizer = request_flags.tokenizer

  inputs = prepare_trt_triton_prompt(
    system_prompt=None,
    prompt=request_flags.prompt,
    tokenizer=tokenizer,
    input_tokens=request_flags.tokens,
    protocol=TritonCommsProtocol.GRPC,
  )

  inputs += prepare_trt_triton_params(
    temperature=request_flags.temperature,
    max_gen_len=request_flags.max_gen_len,
    runtime_top_k=None,
    runtime_top_p=0.9,
    presence_penalty=None,
    repetition_penalty=None,
    stop_words=None,
    bad_words=None,
    protocol=TritonCommsProtocol.GRPC,
  )

  inputs += prepare_trt_triton_stream(
    protocol=TritonCommsProtocol.GRPC,
  )

  outputs = prepare_trt_triton_outputs(
    output_ids=True,
    cum_log_probs=False,
  )

  # create stream?
  if not stream_is_running:
    if stream_collator is None:
      stream_collator = UserDataCollector()
    client.start_stream(callback=partial(callback, stream_collator))
  
  # send request?
  client.async_stream_infer(
    model_name=model_name,
    inputs=inputs,
    request_id=request_flags.request_id,
    outputs=outputs
  )
  
  # wait for server to close stream?
  if not stream_is_running:
    client.stop_stream()
  output_ids: List[np.ndarray] = []

  # max chunk size
  toks_to_gen = request_flags.max_gen_len

  # wait until stream is completed
  while stream_collator._completed_requests.qsize() < toks_to_gen:
    time.sleep(0.05)

  while True:
    try:
      result = stream_collator._completed_requests.get(False)
    except Exception as e:
      break

    if type(result) == InferenceServerException:
      raise result
    else:
      output = result.as_numpy("output_ids")
      output_ids.append(output)

  for i in range(len(output_ids)):
    output_ids[i] = np.reshape(output_ids[i], (-1))
  
  output = np.concatenate(output_ids, axis=0)
  output = output.tolist()
  
  # weird behaviour
  if isinstance(output[0], list):
    print(f"Inside weird behaviour: {output}")
    output = output[0]

  if trim_output_to_eos:
    output = trim_to_eos(output, tokenizer.eos_token)
  
  if output is None:
    raise ValueError("No output")
  
  if request_flags.prompt is not None:
    prompt_tokens = request_flags.prompt.get_ids(tokenizer)
  elif request_flags.tokens is not None:
    prompt_tokens = request_flags.tokens

  full_toks = prompt_tokens + output
  
  if collection_queue is not None:
    thread_id = request_flags.request_id.split("-")[0]
    result_object = (output, full_toks, request_flags.request_id, thread_id)
    collection_queue.put(result_object)
    return
  
  return output, full_toks

def spawn_stream_run_thread(client: tritonclient.grpc.InferenceServerClient, collator: UserDataCollector, request_flags: LLMRequest, thread_id: int, collection_queue: queue.Queue) -> threading.Thread:
  thread_request_flags = replace(request_flags, request_id=f"{request_flags.request_id}-{thread_id}")
  thread = threading.Thread(target=run_stream, args=(client, thread_request_flags, False, True, collator, collection_queue))
  thread.start()
  return thread

def fast_rabin_karp_repeat(output: List[int], min_seq_size: int, max_seq_size: int, min_repeat_proportion) -> bool:
  has_repetion = int_list_has_repeats(output, min_seq_size, max_seq_size, min_repeat_proportion)
  return has_repetion

def get_in_bulk_from_queue(queue: queue.Queue, items: List[Any]):
  while not queue.empty():
    items.append(queue.get())

def run_clients(clients: Dict, stream_collators: Dict, request_flags: LLMRequest):
  threads = []
  result_queue = queue.Queue()
  
  for i in range(len(clients)):
    t = spawn_stream_run_thread(clients[i], stream_collators[i], request_flags, i, result_queue)
    threads.append(t)
  
  for i in range(len(clients)):
    threads[i].join()
  
  all_results = []
  get_in_bulk_from_queue(result_queue, all_results)
  all_results.sort(key=lambda x: x[3])
  return all_results

def evaluate_client_results(all_results: List, request_flags: LLMRequest, eval_has_repeat):
  for i in range(len(all_results)):
    output, full_toks, _, _ = all_results[i]
    if not eval_has_repeat(trim_to_eos(output, request_flags.tokenizer.eos_token)):
      return output, full_toks
  return None, None

def run_clients_with_retries(clients: Dict, stream_collators: Dict, request_flags: LLMRequest, eval_has_repeat, n_retries: int = 3):
  for i in range(n_retries):
    all_results = run_clients(clients, stream_collators, request_flags)
    output, full_toks = evaluate_client_results(all_results, request_flags, eval_has_repeat)
    if output is not None:
      return output, full_toks
  return None, None

def trim_guard_seq(output: List[int], tokenizer: TokenizerBase, trim_eos: bool, rolling_size: int, max_size: int) -> List[int]:  
  if trim_eos:
    output = trim_to_eos(output, tokenizer.eos_token)

  rolling_size += len(output)
  
  if rolling_size > max_size:
    overflow = rolling_size - max_size
    output = output[:-overflow]
  
  return output, rolling_size

def run_chunk_stream(accumulator: StreamChunkAccumulator, request_flags: LLMRequest, chunk_size: int, n_copies: int = 1):
  result_queue = queue.Queue()
  
  max_toks = request_flags.max_gen_len
  toks_so_far = 0

  n_chunks = request_flags.max_gen_len // chunk_size
  n_chunks = n_chunks + 1 if request_flags.max_gen_len % chunk_size != 0 else n_chunks
  request_flags.max_gen_len = chunk_size

  eval_has_repeat = partial(
    fast_rabin_karp_repeat,
    min_seq_size=request_flags.min_repeat_size,
    max_seq_size=request_flags.max_repeat_size,
    min_repeat_proportion=request_flags.min_repeat_proportion)

  # top-level tokenizer
  tokenizer = request_flags.tokenizer

  clients = {i: grpcclient.InferenceServerClient(url=request_flags.url) for i in range(n_copies)}
  stream_collators = {i: UserDataCollector() for i in range(n_copies)}
  
  # start streams
  for i in range(n_copies):
    clients[i].start_stream(callback=partial(callback, stream_collators[i]))

  # [n_chunks - 1] chunks
  for i in range(n_chunks):
    output, full_toks = run_clients_with_retries(clients, stream_collators, request_flags, eval_has_repeat, request_flags.repeat_retries)
    if output is None:
      raise RetryLimitExceeded(f"Retry limit of {request_flags.repeat_retries} exceeded")
    
    request_flags.prompt = None
    request_flags.tokens = full_toks
    
    has_eos = seq_has_eos(output, tokenizer.eos_token)
    output, toks_so_far = trim_guard_seq(output, tokenizer, has_eos, toks_so_far, max_toks)
    accumulator.add_bulk(output)
    if has_eos:
      break

  # close stream
  for i in range(n_copies):
    clients[i].stop_stream()
  
  accumulator.has_finished()
  # print(f"Text: {text}")

class CompletionReason(str, Enum):
  PART_DONE = "stream_part_complete"
  MAX_SIZE = "max_size_reached"

@dataclass
class CompletionResponse:
  content: str
  total_tokens: int
  reason: CompletionReason

  def as_json(self):
    return {
      "content": self.content,
      "reason": self.reason.value,
      "generation_tokens": self.total_tokens,
    }

def decode_token_stream_to_string_chunks(token_stream: Generator, tokenizer: TokenizerBase, chunk_size: int) -> Generator:
  accum_tokens = []
  processed_toks = 0
  convert_to_string_steps = 1
  prev_str = ""
  
  for token in token_stream:
    processed_toks += 1
    accum_tokens = accum_tokens + [token]
    if len(accum_tokens) % convert_to_string_steps == 0:
      new_str = tokenizer.decode(accum_tokens).strip()
      prev_len = len(prev_str)
      prev_str = new_str
      reason =  CompletionReason.PART_DONE if len(accum_tokens) < MAX_SIZE \
                else CompletionReason.MAX_SIZE
      yield CompletionResponse(new_str[prev_len:], 1, reason)

  if processed_toks < len(accum_tokens):
    new_str = tokenizer.decode(accum_tokens).strip()
    prev_len = len(prev_str)
    prev_str = new_str
    reason =  CompletionReason.PART_DONE if len(accum_tokens) < MAX_SIZE \
              else CompletionReason.MAX_SIZE
    yield CompletionResponse(new_str[prev_len:], 1, reason)

@dataclass
class LLMServerInfo:
  url: str
  model_name: str

@dataclass
class Hyperparameters:
  temperature: float
  max_gen_len: int
  runtime_top_k: Optional[int]
  runtime_top_p: Optional[float]
  presence_penalty: Optional[float]
  repetition_penalty: Optional[float]

@dataclass
class AntiRepeatPolicy:
  min_size: int
  max_size: int
  min_repeat_proportion: float
  repeat_retries: int

def make_request_flags(
  tokens: List[int],
  server_info: LLMServerInfo,
  hyperparameters: Hyperparameters,
  anti_repeat_policy: AntiRepeatPolicy,
  tokenizer: TokenizerBase,
  request_id: str,
) -> LLMRequest:
  
  return LLMRequest(
    url=server_info.url,
    model_name=server_info.model_name,
    request_id=str(request_id),
    prompt=None,
    tokens=tokens,
    tokenizer=tokenizer,
    repetition_penalty=hyperparameters.repetition_penalty,
    presence_penalty=hyperparameters.presence_penalty,
    temperature=hyperparameters.temperature,
    max_gen_len=hyperparameters.max_gen_len,
    stop_words=[],
    bad_words=[],
    min_repeat_size=anti_repeat_policy.min_size,
    max_repeat_size=anti_repeat_policy.max_size,
    min_repeat_proportion=anti_repeat_policy.min_repeat_proportion,
    repeat_retries=anti_repeat_policy.repeat_retries,
  )

def decode_token_stream(request_id: int, tokens: List[int], tokenizer: TokenizerBase, server_info: LLMServerInfo, hyperparameters: Hyperparameters, anti_repeat_policy: AntiRepeatPolicy) -> Generator:
  request_flags = make_request_flags(tokens, server_info, hyperparameters, anti_repeat_policy, tokenizer, request_id)
  token_stream_accumulator = StreamChunkAccumulator()
  thread = threading.Thread(target=run_chunk_stream, args=(token_stream_accumulator, request_flags, 16))
  thread.start()
  token_stream = token_stream_accumulator.get()
  for completion_part in decode_token_stream_to_string_chunks(token_stream, tokenizer, 16):
    yield completion_part
  thread.join()


if __name__ == "__main__":
  client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=True)
  
  request_flags = LLMRequest(
    url="localhost:8001",
    model_name="tensorrt_llm",
    request_id="100",
    prompt=Prompt("[INST] Generate a sorting function in C#? [/INST]"),
    tokens=[],
    tokenizer=tokenizer,
    repetition_penalty=None,
    presence_penalty=None,
    temperature=0.9,
    max_gen_len=4096,
    stop_words=[],
    bad_words=[],
    min_repeat_size=2,
    max_repeat_size=8,
    min_repeat_proportion=0.5,
    repeat_retries=3,
  )

  server_info = LLMServerInfo(
    url="localhost:8001",
    model_name="tensorrt_llm",
  )

  hyperparameters = Hyperparameters(
    temperature=0.6,
    max_gen_len=4096,
    runtime_top_k=None,
    runtime_top_p=0.9,
    presence_penalty=None,
    repetition_penalty=None,
  )

  anti_repeat_policy = AntiRepeatPolicy(
    min_size=2,
    max_size=8,
    min_repeat_proportion=0.5,
    repeat_retries=1,
  )

  sys_message = "[INST] <<SYS>>\nAlways assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.\n<</SYS>>\n\n [/INST]"
  ist_message = "[INST] Generate a sorting function in C#? [/INST]"
  # sys_message = tokenizer.tokenize(sys_message, bos=True, eos=False)[0]
  message = sys_message + " " + ist_message
  #message = ist_message

  tokens = Prompt("[INST] Generate a sorting function in C#? [/INST]").get_ids(tokenizer)
  print(tokenizer.decode([31355, 31355, 31355, 31355, 31355]))

  tokens = [1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 2499, 1994, 6985, 411, 2562, 29892, 3390, 29892, 322, 8760, 29889, 2538, 2818, 411, 3477, 3242, 19725, 3447, 11592, 368, 29889, 319, 5405, 10311, 1319, 29892, 443, 621, 936, 29892, 758, 17675, 7612, 29892, 470, 8178, 2793, 29889, 22521, 545, 1634, 3687, 27391, 6534, 2264, 322, 13686, 2068, 29889, 13, 29966, 829, 14816, 29903, 6778, 13, 13, 5631, 403, 263, 16548, 740, 297, 5132, 29973, 518, 29914, 25580, 29962]
  tokens = tokenizer.tokenize(message, bos=True, eos=False)[0]

  for string_chunk in decode_token_stream(100, tokens, tokenizer, server_info, hyperparameters, anti_repeat_policy):
    print(string_chunk.content, flush=True, end="")
    pass
  
