from unakin.llm.adapters.llm_adapter import (AdaptationConfig,
                                             BiasScaleAdaptation,
                                             LMHeadUnfreezeAdaptation,
                                             LoraAdaptation,
                                             NormUnfreezeAdaptation,
                                             SoftPromptAdaptation)


class AdapterConfigFactory:
  
  @staticmethod
  def from_file(path: str) -> AdaptationConfig:
    if path.endswith(".json"):
      return AdapterConfigFactory.from_json(path)

  @staticmethod
  def from_json(json_path_or_dict: str) -> AdaptationConfig:
    import json
    
    if isinstance(json_path_or_dict, str):
      with open(json_path_or_dict, "r") as f:
        json_dict = json.load(f)
    else:
      json_dict = json_path_or_dict
    
    if "lora" in json_dict:
      lora = LoraAdaptation(**json_dict["lora"])
    else:
      lora = None
    if "bias_scale" in json_dict:
      bias_scale = BiasScaleAdaptation(**json_dict["bias_scale"])
    else:
      bias_scale = None
    if "norm" in json_dict:
      norm = NormUnfreezeAdaptation(**json_dict["norm"])
    else:
      norm = None
    if "soft_prompt" in json_dict:
      soft_prompt = SoftPromptAdaptation(**json_dict["soft_prompt"])
    else:
      soft_prompt = None
    if "lm_head" in json_dict:
      lm_head = LMHeadUnfreezeAdaptation(**json_dict["lm_head"])
    else:
      lm_head = None

    return AdaptationConfig(
      lora=lora,
      bias_scale=bias_scale,
      norm=norm,
      soft_prompt=soft_prompt,
      lm_head=lm_head
    )