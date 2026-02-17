from typing import List, Literal, TypedDict
from unakin.llm.tokenizers import LlamaTokenizer

# TODO should be imported from official llama repo? -------------

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]

# ---------------------------------------------------------------

# TODO will be sent by the server
MAX_SEQUENCE_LENGTH = 8192

# in case none supplied by user, use default system prompt:
DEFAULT_SYSTEM_PROMPT = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Ensure replies promote fairness and positivity."

def make_input_tokens(messages: List[Message], tokenizer: LlamaTokenizer) -> List[int]:
    assert len(messages) > 0, "No messages supplied!"

    # TODO no handling for UNSAFE portions yet

    prompt_tokens = []
    is_max_length_exceeded = True
 
    while is_max_length_exceeded:
        if messages[0]["role"] == "system":
            dialog = [
                {
                    "role": messages[1]["role"],
                    "content": B_SYS
                    + messages[0]["content"]
                    + E_SYS
                    + messages[1]["content"],
                }
            ] + messages[2:]

            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )

        try:
            len(dialog)
        except Exception:
            dialog = messages

        dialog_tokens: List[int] = sum(
            [
                tokenizer.tokenize(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                    bos=True,
                    eos=True,
                )
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.tokenize(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            bos=True,
            eos=False,
        )

        # TODO?? flatten list of lists dialog_tokens into a single list of int
        dialog_tokens = [
            token
            for sublist in dialog_tokens
            for token in sublist
        ]

        prompt_tokens = dialog_tokens

        if len(prompt_tokens) < MAX_SEQUENCE_LENGTH:
            is_max_length_exceeded = False
            break
        else:
            # remove the first user / assistant message pair from the chat
            if messages[0]["role"] == "system":
                if len(messages) > 3:
                    messages = messages[0] + messages[3:]
                else:
                    raise Exception("Not enough messages in the dialog and max token length exceeded")
            else:
                if len(messages) > 1:
                    messages = messages[1:]
                else:
                    raise Exception("Not enough messages in the dialog and max token length exceeded")

    return prompt_tokens


if __name__ == "__main__":

    TOKENIZER_PATH = "/home/ubuntu/codellama_ckpt/CodeLlama-13b-Instruct/tokenizer.model"
    tokenizer = LlamaTokenizer(model_path=TOKENIZER_PATH)

    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]

    print(tokenizer.decode(make_input_tokens(messages=messages, tokenizer=tokenizer)))
