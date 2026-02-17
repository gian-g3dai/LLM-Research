import torch
from torch import Tensor
from transformers import AutoModel

from unakin.search.models.base import BaseSearchModel


class AttentionMeanPooling(torch.autograd.Function):
  @staticmethod
  def forward(ctx, inputs: Tensor, mask: Tensor) -> Tensor:
    input_mask_expanded = mask.unsqueeze(-1)
    input_mask_expanded = input_mask_expanded.expand(inputs.size()).float()
    
    weighted_sum  = torch.sum(inputs * input_mask_expanded, dim=1)
    pooled_states = weighted_sum / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    pooled_states = torch.nn.functional.normalize(pooled_states, p=2, dim=1)
    return pooled_states

class SbertSearchModel(BaseSearchModel):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.model = AutoModel.from_pretrained('flax-sentence-embeddings/stackoverflow_mpnet-base')

  def encode_query(self,
                   query: Tensor,
                   attention_mask: torch.Tensor | None = None) -> Tensor:
    query_outs = self.forward(query, attention_mask)
    return query_outs
  
  def encode_document(self,
                      document: Tensor,
                      attention_mask: torch.Tensor | None = None) -> Tensor:
    document_outs = self.forward(document, attention_mask)
    return document_outs
  
  def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
      mask = torch.ones(inputs.shape[0], inputs.shape[1],
                         device=inputs.device, dtype=torch.int32)

    model_outs = self.model(inputs, attention_mask=mask)
    hidden_states = model_outs[0]
    outputs = AttentionMeanPooling.apply(hidden_states, mask)
    return outputs    
    
  def embedding_size(self) -> int:
    return self.model.config.hidden_size
  
  def support_onnx(self) -> bool:
    return True
