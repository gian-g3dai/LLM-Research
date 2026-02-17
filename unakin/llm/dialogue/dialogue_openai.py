import openai

from typing import List, Optional

from tqdm import tqdm

from unakin.llm.dialogue.base import DialogueModelBase, ChatPrediction, Dialog


class DialogueModelOpenAI(DialogueModelBase):
    
    def __init__(self, api_key, model_name="gpt-4"):
        self.api_key = api_key
        openai.api_key = api_key
        self.model_name = model_name
        
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        
        kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repeat_penalty - 1.0,
        }

        if max_gen_len is not None:
            kwargs["max_tokens"] = max_gen_len
        
        completions = [
            openai.ChatCompletion.create(
                model=self.model_name,
                messages=d,
                
            )
            for d in tqdm(dialogs, total=len(dialogs))
        ]

        # print first 30 characters of each completion
        for c in completions:
            print(c["choices"][0]["message"]["content"][:80])

        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": response["choices"][0]["message"]["content"],
                }
            }
            for response in completions
        ]
