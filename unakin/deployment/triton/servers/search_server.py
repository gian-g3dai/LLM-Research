import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import onnxruntime as rt
from onnxruntime.transformers import optimizer as onnx_trf_optim

import torch
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton

from unakin.search.models.base import BaseSearchModel
from unakin.search.models.utils import SearchModelFactory
import tensorrt as trt

@dataclass
class SearchServerConfig:
  # replica specifics
  n_replicas: int

  # model speciffics
  model_config_or_path: str
  device: str = "cpu"
  run_in_half: bool = True  

  # onnx specifics
  export_to_onnx: bool = False
  
  # triton specifics
  batching: bool = True
  max_batch_size: int = 1024
  response_cache: bool = False

  # batch queue specifics
  max_queue_delay_ms: int = 1000
  preserve_ordering: bool = True
  preferred_batch_size: List[int] = field(default_factory=lambda: [256, 512, 1024])

  # connectivity specifics
  port: int = 8000

  @classmethod
  def from_json(cls, file_path_or_config: str | dict):
    if isinstance(file_path_or_config, str):
      with open(file_path_or_config, "r") as f:
        json_config = json.load(f)
    else:
      json_config = file_path_or_config

    return cls(**json_config)

  def get_triton_model_config(self) -> ModelConfig:
    batcher = DynamicBatcher(
      max_queue_delay_microseconds=self.max_queue_delay_ms * 1,
      preserve_ordering=self.preserve_ordering,
      preferred_batch_size=self.preferred_batch_size 
    )

    config = ModelConfig(
      batching=self.batching,
      max_batch_size=self.max_batch_size,
      batcher=batcher,
      response_cache=self.response_cache,
    )

    return config 

class SearchServerInferenceEngine:
  def __init__(self, config: SearchServerConfig, reset: bool) -> None:
    self.model, self.tokenizer = SearchModelFactory.from_json(config.model_config_or_path)
    self.model.to(config.device)
    self.model.eval()
    self.embed_dim = self.model.embedding_size()
    self.config = config
    self.reset = reset

    # very important, because for onnx mixed precision we
    # need the original model
    if config.run_in_half and not config.export_to_onnx:
      self.model = self.model.half()

    # export to onnx
    if config.export_to_onnx:
      if self.model.support_onnx():
        self._export_to_onnx("sbert.onnx")
        self._load_onnx("sbert.onnx")
        self._is_onnx = True
    else:
      self._is_onnx = False

    self.device = config.device

  def _load_onnx(self, file_path: str) -> None:
    # first delete the model
    del self.model

    # clear torch cache
    torch.cuda.empty_cache()


    session_options = rt.SessionOptions()
    session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    self.model = rt.InferenceSession(file_path,
                                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                                     provider_options=[{}, {}],)

  def _export_to_onnx(self, file_path: str) -> None:
    # dummy input
    if not self.reset:
      return

    dummy_input = torch.ones(self.config.max_batch_size, 512, device=self.config.device, dtype=torch.int32)
    dummy_mask = torch.ones(self.config.max_batch_size, 512, device=self.config.device, dtype=torch.int32)
    dummy_input = (dummy_input, dummy_mask)

    # export
    torch.onnx.export(
      self.model, dummy_input, file_path,
      input_names=["input_ids", "attention_mask"],
      output_names=["output"],
      # only batch size is dynamic
      dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "output": {0: "batch_size"}
      },
      opset_version=17, do_constant_folding=True, verbose=True
    )

    # re-export
    if self.config.run_in_half:
      
      opt_options = onnx_trf_optim.FusionOptions(model_type="bert")
      opt_options.enable_embed_layer_norm = False
      opt_options.enable_gelu_approximation = True
      opt_options.use_multi_head_attention = True
      
      optimized_model = onnx_trf_optim.optimize_model(
        file_path, model_type="bert",
        num_heads=12, hidden_size=768,
        optimization_options=opt_options)
      
      optimized_model.convert_float_to_float16()
      optimized_model.save_model_to_file(file_path)

    # remove to release memory
    del dummy_input 

  def _prepare_input(self, strings: np.ndarray) -> np.ndarray | torch.Tensor:
    # convert from string to ids
    sentences = np.char.decode(strings.astype("bytes"), "utf-8")
    sentences = sentences.tolist()
    if isinstance(sentences[0], list):
      sentences = [sentence[0] for sentence in sentences]

    print(sentences)

    # tokenize
    return_format = "np" if self._is_onnx else "pt"
    tokenizer_outputs = self.tokenizer.encode(sentences, return_tensors=return_format)
    ids, mask = tokenizer_outputs.ids, tokenizer_outputs.attention_mask
    if return_format == "pt":
      ids, mask = ids.to(self.device), mask.to(self.device)
    elif return_format == "np":
      ids, mask = ids.astype(np.int32), mask.astype(np.int32)
    return ids, mask 

  def _prepare_output(self, embeddings: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(embeddings, torch.Tensor):    
      if self.config.run_in_half:
        embeddings = embeddings.float()
      return embeddings.cpu().detach().numpy()

    return embeddings

  @batch
  def encode_query(self, query: np.ndarray) -> np.ndarray:
    encode_query_start_time = time.time()
    with torch.inference_mode():
      ids, mask = self._prepare_input(query)
      inf_fn = BaseSearchModel.onnx_encode_query if self._is_onnx else \
               BaseSearchModel.torch_encode_query
      print(ids.shape, mask.shape)
      embeddings = inf_fn(self.model, ids, mask)
    response = self._prepare_output(embeddings)
    print(f"Encode query time: {time.time() - encode_query_start_time}")
    return [response]
    
  @batch
  def encode_document(self, document: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
      ids, mask = self._prepare_input(document)
      inf_fn = BaseSearchModel.onnx_encode_query if self._is_onnx else \
               BaseSearchModel.torch_encode_query
      embeddings = inf_fn(self.model, ids, mask)
    return [self._prepare_output(embeddings)]


def launch_server(config: SearchServerConfig):
  n_replicas = config.n_replicas
  engines = []
  
  for i in range(n_replicas):
    is_main_replica = i == 0
    engine = SearchServerInferenceEngine(config, reset=is_main_replica)
    engines.append(engine)
  
  engine_embed_dim = engine.embed_dim

  with Triton() as triton:
    output_dtype = np.float32 # because of transformer optimization

    triton.bind(
      model_name="query-sentence-embeddings",
      infer_func=[e.encode_query for e in engines],
      inputs=[Tensor(name="query", dtype=bytes, shape=(1,))],
      outputs=[Tensor(name="response", dtype=output_dtype, shape=(engine_embed_dim,))],
      config=config.get_triton_model_config(),
      strict=True
    )

    triton.bind(
      model_name="document-sentence-embeddings",
      infer_func=[e.encode_document for e in engines],
      inputs=[Tensor(name="document", dtype=bytes, shape=(1,))],
      outputs=[Tensor(name="response", dtype=output_dtype, shape=(engine_embed_dim,))],
      config=config.get_triton_model_config(),
    )

    print(config.get_triton_model_config())
    triton.serve()

__DEV_CONFG__ = SearchServerConfig(
    model_config_or_path={
      "model": {
        "type": "sbert",
      },
      "tokenizer": {
        "type": "sbert",
      }
    },
    export_to_onnx=True,
    n_replicas=2,
    device="cuda:0",
    run_in_half=True,
    batching=True,
    max_batch_size=32,
    response_cache=False,
    max_queue_delay_ms=10,
    preserve_ordering=True,
    preferred_batch_size=[32, 16, 8, 4],
    port=8000
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True, help="Path to config file")
  parser.add_argument("--develop", action="store_true", help="Run in development mode")
  args = parser.parse_args()

  if args.develop:
    config = __DEV_CONFG__
  else:
    config = SearchServerConfig.from_json(args.config)

  launch_server(config)