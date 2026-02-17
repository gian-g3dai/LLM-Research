from abc import abstractmethod

import numpy as np
import onnxruntime as rt
import torch
import tensorrt as trt 


class BaseSearchModel(torch.nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def forward(self, *args, **kwargs) -> None:
    raise NotImplementedError
  
  @abstractmethod
  def encode_query(self, query: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
    raise NotImplementedError
  
  @abstractmethod
  def encode_document(self, document: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
    raise NotImplementedError
  
  @abstractmethod
  def embedding_size(self) -> int:
    raise NotImplementedError
  
  @abstractmethod
  def support_onnx(self) -> bool:
    raise NotImplementedError

  @staticmethod
  def onnx_encode_query(runtime: rt.InferenceSession,
                        query: np.ndarray,
                        attn_mask: np.ndarray | None = None) -> np.ndarray:
    if attn_mask is None:
      attn_mask = np.ones((query.shape[0], query.shape[1]), dtype=np.int64)
    
    inputs = {"input_ids": query, "attention_mask": attn_mask}
    
    outputs = runtime.run(None, inputs)
    return outputs[0]
  
  @staticmethod
  def onnx_encode_document(runtime: rt.InferenceSession,
                           document: torch.Tensor,
                           attn_mask: torch.Tensor | None = None) -> torch.Tensor:
    if attn_mask is None:
      attn_mask = np.ones((document.shape[0], document.shape[1]), dtype=np.int32)
    
    inputs = {
      "input_ids": document,
      "attention_mask": attn_mask
    }

    print(type(inputs["input_ids"]))
    
    outputs = runtime.run(None, inputs)
    return outputs[0]
  
  def torch_encode_query(model: 'BaseSearchModel',
                         query: torch.Tensor,
                         attn_mask: torch.Tensor | None = None) -> torch.Tensor:
    return model.encode_query(query, attn_mask)
  
  def torch_encode_document(model: 'BaseSearchModel',
                            document: torch.Tensor,
                            attn_mask: torch.Tensor | None = None) -> torch.Tensor:
    return model.encode_document(document, attn_mask)