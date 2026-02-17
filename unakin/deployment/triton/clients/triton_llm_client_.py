from dataclasses import dataclass
from enum import Enum
from functools import partial
import threading
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonclient
import tritonclient.grpc
from tritonclient.utils import np_to_triton_dtype

from multiprocessing import Queue
from typing import Any, List, Optional
import random
import numpy as np
import queue
import time

from unakin.llm.tokenizers.base import TokenizerBase
from unakin.llm.tokenizers.llama_tokenizer import LlamaTokenizer

TOKENIZER_PATH = "/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model"
TOKENIZER = LlamaTokenizer(model_path=TOKENIZER_PATH)

# make a singleton END_OF_STREAM object
class StreamControlMessage(Enum):
  END_OF_STREAM = "END_OF_STREAM"

class TritonCommsProtocol(str, Enum):
  HTTP = "http"
  GRPC = "grpc"

class CompletionReason(str, Enum):
  PART_DONE = "stream_part_complete"
  MAX_SIZE = "max_size_reached"

@dataclass
class StreamItem:
  content: str
  total_tokens: int
  reason: CompletionReason

  def as_json(self):
    return {
      "content": self.content,
      "reason": self.reason.value,
      "generation_tokens": self.total_tokens,
    }

class StreamQueue:
  def __init__(self, tokenizer: TokenizerBase, stream_id: int = None) -> None:
    self.tokenizer = tokenizer
    self._completed_requests = Queue()
    self._results = []
    self._stream_id = stream_id
    
    # for EOS intercept 
    self.done = False

  def put(self, a: Any) -> None:
    self._completed_requests.put(a)
  
  def get(self, block: bool) -> Any:
    return self._completed_requests.get(block=block)

  def set_id(self, stream_id: int) -> None:
    self._stream_id = stream_id

  def mark_finished(self) -> None:
    self.done = True

  def has_finished(self) -> bool:
    return self.done

  def reset(self) -> None:
    self._completed_requests = Queue()
    self._results = []
    self.done = False

  def __iter__(self):
    item = None
    
    string = ""
    prev_string = ""
    full_output_ids = []
    
    while item != StreamControlMessage.END_OF_STREAM:
      item = self.get(block=True)
      
      if item is StreamControlMessage.END_OF_STREAM: 
        break

      elif item is None:
        time.sleep(0.05)

      else:
        full_output_ids.extend(item)
        string = self.tokenizer.decode(full_output_ids).strip()
        new_chunk = string[len(prev_string):]
        prev_string = string

        stream_item = StreamItem(
          content=new_chunk,
          total_tokens=len(full_output_ids),
          reason=CompletionReason.PART_DONE,
        )

        yield stream_item
      
def stream_accumulate_callback(
  q: StreamQueue,
  result: tritonclient.grpc.InferResult,
  error: str
) -> None:
  if error:
    q.put(error)
  else:
    if result.get_output("output_ids") is not None:
      output_ids = result.as_numpy("output_ids")
      output_ids = output_ids.tolist()
      # stream returns only (1, 1, ?) shapes      
      output_ids = output_ids[0][0]
      if TOKENIZER.eos_token in output_ids:
        pass
      q.put(output_ids)
    else:
      raise ValueError("No output_ids in result")

class RequestIdManager:
  def __init__(
    self,
    capacity: int = 1_000_000,
    expiration_minutes: int = 120,
  ) -> None:
    self._capacity = capacity
    self._expiration_minutes = expiration_minutes
    self._request_ids = {}

  def is_valid(self, req_id: int | None) -> bool:    
    # new streams
    if req_id is None:
      return False

    # new streams   
    if req_id not in self._request_ids:
      return True
    
    now = time.time()
    id_time = self._request_ids[req_id]
    
    # expired streams
    if now - id_time > self._expiration_minutes * 60:
      del self._request_ids[req_id]
      return False
    
    # update time to keep it from expiring
    self._request_ids[req_id] = now
    return True
  
  def allocate_id(self) -> int:
    new_id = random.randint(0, self._capacity)
    while new_id in self._request_ids:
      new_id = random.randint(0, self._capacity)
    self._request_ids[new_id] = time.time()
    return new_id

class TritonLLMClient:
  
  def __init__(
    self,
    url: str = "localhost:8001",
    model_name: str = "tensorrt_llm",
    tokenizer_eos_id: int = 2,
  ) -> None:
    
    self.url = url
    self.model_name = model_name
    self.tokenizer_eos_id = tokenizer_eos_id
    self.req_id_manager = RequestIdManager(expiration_minutes=120)
  
  def prepare_triton_tensor(
    self,
    key: str,
    value: np.ndarray,
    protocol: TritonCommsProtocol):

    client_util = httpclient if protocol == TritonCommsProtocol.HTTP\
                  else grpcclient
    
    tensor = client_util.InferInput(key, value.shape, np_to_triton_dtype(value.dtype))
    tensor.set_data_from_numpy(value)
    return tensor

  def _predict(
      self,
      batch_tokens: List[List[int]],
      stream_queue: StreamQueue,
      seed: List[int] = None
    ):

    if seed is None:
      seed = [1 for _ in range(len(batch_tokens))]
    else:
      assert len(seed) == len(batch_tokens), "Seed must be same length as batch_tokens"

    bsz = len(batch_tokens)    
    input_leghts = [len(sample_tokens) for sample_tokens in batch_tokens]
    max_length = max(input_leghts)

    # Pad to max length
    batch_token_data = [
      sample_tokens + [self.tokenizer_eos_id] * (max_length - len(sample_tokens))
      for sample_tokens in batch_tokens
    ]

    batch_token_data = np.array(batch_token_data, dtype=np.int32)
    batch_lengths_data = np.array(input_leghts, dtype=np.int32)\
                           .reshape(bsz, 1)
    
    batch_end_ids = np.array([self.tokenizer_eos_id], dtype=np.int32)\
                      .repeat(bsz, axis=0)\
                      .reshape(bsz, 1)

    batch_log_probs = np.array([True], dtype=np.bool_)\
                        .repeat(bsz, axis=0)\
                        .reshape(bsz, 1)
    
    batch_random_seeds = np.array(seed, dtype=np.uint64)\
                           .reshape(bsz, 1)


    inputs = [
      self.prepare_triton_tensor("random_seed", batch_random_seeds, TritonCommsProtocol.GRPC),
      self.prepare_triton_tensor("input_ids", batch_token_data, TritonCommsProtocol.GRPC),
      self.prepare_triton_tensor("input_lengths", batch_lengths_data, TritonCommsProtocol.GRPC),
      self.prepare_triton_tensor("end_id", batch_end_ids, TritonCommsProtocol.GRPC),
    ]

    batch_max_gen_lengths = np.array([2048], dtype=np.int32)\
                              .repeat(bsz, axis=0)\
                              .reshape(bsz, 1)

    batch_streaming = np.array([True], dtype=np.bool_)\
                        .repeat(bsz, axis=0)\
                        .reshape(bsz, 1)  
    
    batch_temperature = np.array([0.6], dtype=np.float32)\
                          .repeat(bsz, axis=0)\
                          .reshape(bsz, 1)

    inputs += [
      # self.prepare_triton_tensor("beam_width", batch_beam_width, TritonCommsProtocol.GRPC),
      self.prepare_triton_tensor("temperature", batch_temperature, TritonCommsProtocol.GRPC),
      self.prepare_triton_tensor("request_output_len", batch_max_gen_lengths, TritonCommsProtocol.GRPC),
      self.prepare_triton_tensor("streaming", batch_streaming, TritonCommsProtocol.GRPC),
    ]
    
    client = tritonclient.grpc.InferenceServerClient(
      url=self.url,
      verbose=False,
    )

    outputs = [
      tritonclient.grpc.InferRequestedOutput("output_ids"),
    ]

    # get the stream id
    stream_req_id = None
    # in theory this should help with speed?
    if not self.req_id_manager.is_valid(stream_req_id):
      stream_req_id = self.req_id_manager.allocate_id()
      stream_queue.set_id(stream_req_id)

    client.start_stream(partial(stream_accumulate_callback, stream_queue))
    client.async_stream_infer("tensorrt_llm", inputs, outputs=outputs, request_id=f"{stream_req_id}")
    # I have no idea why this block exists here
    """
    while True:
      try:
        result = q.get(block=False)
      except Exception as e:
        print(e)
        break
    """

    # TODO: This is a blocking set of calls  
    client.stop_stream()
    client.close()
    stream_queue.put(StreamControlMessage.END_OF_STREAM)

    # end of stream confirmation
    print(f"Got here")
    print(f"Finished streaming {bsz} samples")

  def predict(
    self,
    batch_tokens: List[List[int]],
    n_seeds: Optional[int] = None,
    reuse_streams: List[Optional[StreamQueue]] = None,
  ) -> List[StreamQueue]:    
    assert len(batch_tokens) == 1, "Batch size must be 1"
    streams = []

    if not n_seeds:
      n_seeds = 1

    if reuse_streams:
      assert len(reuse_streams) == n_seeds, "n_seeds must match reuse_stream length"

    for seed_idx in range(n_seeds):
      stream = StreamQueue(tokenizer=TOKENIZER) \
               if not reuse_streams \
               else reuse_streams[seed_idx]
      streams.append(stream)
      threading.Thread(
        target=self._predict,
        args=(batch_tokens, stream, [seed_idx])).start()
    
    return streams
  
  def batch_predict(
    self,
    batch_tokens: List[List[int]],
    n_seeds: Optional[int] = None,
  ):
    batch_streams = []

    for batch_idx in range(len(batch_tokens)):
      sample_tokens = batch_tokens[batch_idx]
      sample_streams = self.predict([sample_tokens], n_seeds=n_seeds)
      batch_streams.append(sample_streams)

    return batch_streams

if __name__ == "__main__":
  client = TritonLLMClient()

  instruction_1 = "[INST] Tell me a story. [/INST]"
  batch_tokens = TOKENIZER.tokenize(instruction_1, bos=True, eos=False)
  # batch_tokens = [batch_tokens[0] for _ in range(2)]
  streams = client.predict(batch_tokens, n_seeds=1)
  output_ids = []
  for r in streams[0]:
    print(r.content, end="", flush=False)
  print(streams[0].done)

  