from unakin.llm._data.datasources.train_sample import LMTrainingBatch
from unakin.llm.tokenizers.base import TokenizerBase
from unakin.llm.metrics.base import BaseMetric, MetricValue, MetricType
from unakin.llm.metrics.utils import to_metric_value, aggregate_dictionaries, flatten_dict, partition_by_index_mask
from typing import List, Tuple, Dict, Any, Optional, Union
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np



class PerplexityMetric(BaseMetric):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def evaluate_step(self, model: torch.nn.Module, batch: LMTrainingBatch, tokenizer: Optional[TokenizerBase]) -> Dict:
    input_ids, output_ids, output_mask, position_ids, seq_lens, datasource_index_mask = self.unpack_batch(batch)

    seq_lens = seq_lens[0]

    # ignore index for entropy
    output_ids[output_mask == 0] = -100

    preds: torch.Tensor = model.forward(input_ids, position_ids) #, seq_lens)

    bsz, seq_len, _ = preds.shape
    preds = preds.view(bsz * seq_len, -1)
    output_ids = output_ids.view(bsz * seq_len)

    loss_per_token = F.cross_entropy(preds, output_ids, ignore_index=-100, reduction="none")

    loss_per_token = loss_per_token.view(bsz * seq_len)

    pred_tokens = torch.argmax(preds, dim=-1)
    
    pred_tokens = pred_tokens.view(bsz * seq_len)
    output_mask = output_mask.view(bsz * seq_len)

    valid_output_mask = output_mask == 1

    valid_preds = pred_tokens[valid_output_mask]
    valid_tokens = output_ids[valid_output_mask]

    datasource_index_mask = datasource_index_mask.view(bsz * seq_len)

    precision_per_datasource = partition_by_index_mask(valid_preds == valid_tokens, datasource_index_mask[valid_output_mask])
    loss_per_datasource = partition_by_index_mask(loss_per_token[valid_output_mask], datasource_index_mask[valid_output_mask])
  
    loss_per_datasource = {idx.item(): loss.mean() for idx, loss in loss_per_datasource.items()}
    ppl_per_datasource = {idx: 2 ** loss for idx, loss in loss_per_datasource.items()}
    precision_per_datasource = {idx.item(): precision.float().mean() for idx, precision in precision_per_datasource.items()}

    ret_dict = {
      "loss": loss_per_datasource,
      "perplexity": ppl_per_datasource,
      "precision": precision_per_datasource
    }

    del preds, output_ids, output_mask, valid_preds, valid_tokens, valid_output_mask, loss_per_token, pred_tokens
    torch.cuda.empty_cache()
    return ret_dict, len(seq_lens)

  def evaluate_loader(self,
                      loader: torch.utils.data.DataLoader,
                      model: torch.nn.Module,
                      pbar,
                      tokenizer: Optional[TokenizerBase],
                      datasource_names: List[str]) -> List[MetricValue]:
    model.eval()
    step_dicts = []
    with torch.no_grad():

      for batch in loader:

        metrics, n_samples = self.evaluate_step(model, batch, tokenizer)
        
        step_dicts.append(metrics)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
          pbar.update(n_samples)

    model.train()

    agg_metrics = aggregate_dictionaries(step_dicts)

    # convet ds indices to names
    for metric, datasource_split in agg_metrics.items():
      agg_metrics[metric] = {datasource_names[ds_idx]: value for ds_idx, value in datasource_split.items()}

    flat_metrics = flatten_dict(agg_metrics)

    metric_values = [to_metric_value(metric_name, value) for metric_name, value in flat_metrics.items()]

    return metric_values

  def unpack_batch(self, batch: LMTrainingBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.cuda.current_device()
    inputs_ids, output_ids, output_mask, position_ids, seq_lens, datasource_index_mask = batch
    inputs_ids: torch.Tensor = inputs_ids.to(device)
    output_ids: torch.Tensor = output_ids.to(device)
    output_mask: torch.Tensor = output_mask.to(device)
    position_ids: torch.Tensor = position_ids.to(device)
    datasource_index_mask: torch.Tensor = datasource_index_mask.to(device)
    return inputs_ids, output_ids, output_mask, position_ids, seq_lens, datasource_index_mask