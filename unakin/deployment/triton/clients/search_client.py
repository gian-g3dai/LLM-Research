import concurrent.futures
import json
from concurrent.futures import Future
from dataclasses import dataclass
from typing import List

import numpy as np
from pytriton.client import FuturesModelClient


@dataclass
class SearchClientConfig:
  url: str = "localhost"
  port: int = 8000
  model_name: str = "sentence-embeddings"
  model_version: int = -1
  max_concurrency: int = 8192
  verbose: bool = False

  @classmethod
  def from_json(cls, file_path_or_config: str | dict):
    if isinstance(file_path_or_config, str):
      with open(file_path_or_config, "r") as f:
        json_config = json.load(f)
    else:
      json_config = file_path_or_config

    return cls(**json_config)

  def get_query_triton_client(self) -> FuturesModelClient:
    return FuturesModelClient(
      url=f"http://{self.url}:{self.port}",
      model_name=f"query-{self.model_name}",
      max_workers=self.max_concurrency,
    )
  
  def get_document_triton_client(self) -> FuturesModelClient:
    return FuturesModelClient(
      url=f"http://{self.url}:{self.port}",
      model_name=f"document-{self.model_name}",
      max_workers=self.max_concurrency,
    )
  
class TritonSearchClient:
  def __init__(self, config: SearchClientConfig) -> None:
    self.config = config
  
  def _process_future(self, future: Future) -> np.ndarray:
    result = future.result()
    result = result["response"] 
    return result

  def encode_query(self, text: str | List[str], batch_size: int = 256) -> np.ndarray:
    if isinstance(text, str):
      text = [text]

    batch_slices = []
    for i in range(0, len(text), batch_size):
      batch_slices.append(text[i:i+batch_size])

    with self.config.get_query_triton_client() as client:
      futures: List[Future] = []
      for text_batch in batch_slices:
        batch_start_time = time.time()
        use_batch = len(text_batch) > 1
        infer_fn = client.infer_sample if not use_batch else client.infer_batch
        # this is required for `infer_batch`
        if use_batch:
          text_batch = list(map(lambda x: [x], text_batch))
        
        np_text = np.array(text_batch)
        np_text = np.char.encode(np_text, "utf-8")
        futures.append(infer_fn(np_text))
        print(f"Batch time: {time.time() - batch_start_time}")

    future_wait_time = time.time()
    done_futures, _ = concurrent.futures.wait(futures)
    print(f"Future wait time: {time.time() - future_wait_time}")
    results = list(map(self._process_future, done_futures))    

    concat_time = time.time()
    # stack into a single array
    if results[0].ndim == 1:
      results = np.stack(results, axis=0)
    else:
      results = np.concatenate(results, axis=0)
    print(f"Concat time: {time.time() - concat_time}")
    return results
  
  def encode_document(self, text: str | List[str], batch_size: int = 256) -> np.ndarray:
    if isinstance(text, str):
      text = [text]
    text = [np.char.encode(t, "utf-8") for t in text]

    batch_slices = []
    for i in range(0, len(text), batch_size):
      batch_slices.append(text[i:i+batch_size])

    with self.config.get_query_triton_client() as client:
      futures: List[Future] = []
      for text_batch in batch_slices:
        use_batch = len(text_batch) > 1
        infer_fn = client.infer_sample if not use_batch else client.infer_batch
        # this is required for `infer_batch`
        if use_batch:
          text_batch = list(map(lambda x: [x], text_batch))
        
        np_text = np.array(text_batch)
        np_text = np.char.encode(np_text, "utf-8")
        futures.append(infer_fn(np_text))

    results = list(map(self._process_future, futures))

    # stack into a single array
    if results[0].ndim == 1:
      results = np.stack(results, axis=0)
    else:
      results = np.concatenate(results, axis=0)
  
    return results

if __name__ == "__main__":
  import time
  url = "localhost"
  #url = "34.173.70.1"
  client = TritonSearchClient(SearchClientConfig(url=url))
  start_time = time.time()
  with open("search_client.py", "r") as f:
    text = f.read()
  print(type(client.encode_query([text] * 2048, batch_size=32)))
  print(time.time() - start_time)