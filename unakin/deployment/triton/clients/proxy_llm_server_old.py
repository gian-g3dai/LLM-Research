import asyncio
import json
import random
import threading
from typing import List, Literal, TypedDict
import time
import datetime

import fastapi
import starlette
import uvicorn
from fastapi import FastAPI, WebSocket
from dataclasses import replace

from unakin.deployment.triton.clients.llm_client import (
    AntiRepeatPolicy,
    Hyperparameters,
    LLMServerInfo,
    decode_token_stream,
    RetryLimitExceeded,
    InferenceServerException,
    CompletionResponse
)
from unakin.llm.tokenizers import LlamaTokenizer

from auth_connector import UserResponse, AuthServiceConnector

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

TOKENIZER_PATH = "/home/ubuntu/codellama_ckpt/13B_tokenizer/tokenizer.model"
tokenizer = LlamaTokenizer(model_path=TOKENIZER_PATH)

END_OF_GENERATOR = object()

DEFAULT_SYSTEM_PROMPT = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Ensure replies promote fairness and positivity."

app = FastAPI()

auth_connector = AuthServiceConnector(
    auth_server_url="https://api.unakin.ai/auth",
    frontend_url="http://dashboar.unakin.ai"
)

app.include_router(auth_connector, prefix="/auth", tags=["Authentication"])

class TimeoutException(Exception):
    pass

class InternalServerError(Exception):
    pass

def run_generator(gen, q: asyncio.Queue):
    def worker():
        for item in gen:
            q.put_nowait(item)  # Non-blocking put operation
        q.put_nowait(END_OF_GENERATOR)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=60)

    if thread.is_alive():
        raise TimeoutException("Generator thread timed out")    

def make_input_tokens(messages: List[Message], method: str = "ours") -> List[int]:
    assert len(messages) > 0, "No messages supplied!"

    # TODO no handling for UNSAFE portions yet

    prompt_tokens = []
    is_max_length_exceeded = True

    # TODO: @Vlad change this
    # There are no longer different ways of tokenizing. At least for now.
    # We might think of something in the future when, can be modularised, but realistically there
    # will be no branches for the foreseable future.
    if False:

        while is_max_length_exceeded:
            dialog_tokens: List[int] = []
            dialog = messages

            if dialog[0]["role"] == "system":
                dialog_tokens += tokenizer.tokenize(f"{B_INST} {B_SYS} {dialog[0]['content']} {E_SYS} {E_INST}")
                dialog = dialog[1:]
            else:
                dialog_tokens += tokenizer.tokenize(f"{B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT} {E_SYS} {E_INST}")

            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )

            conv_tokens: List[int] = sum(
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
            conv_tokens += tokenizer.tokenize(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )

            dialog_tokens += conv_tokens

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
    
    # TODO: change this as well
    elif True:
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
        
    else:
        raise Exception("Unknown tokenization method")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # we implement header parsing by hand since https://github.com/tiangolo/fastapi/pull/10147 is not ready yet
    # discussion here: https://github.com/tiangolo/fastapi/discussions/8983

    authorization_header = websocket.headers.get("Authorization", None)

    if not authorization_header or not authorization_header.startswith("Bearer "):
      raise fastapi.HTTPException(
          status_code=fastapi.status.HTTP_403_FORBIDDEN,
          detail="No authorization provided!")
    
    token = authorization_header.split("Bearer ")[1]
    # get the user from the token
    current_user = await auth_connector.r_get_current_user(token=token)
    # TODO check user has chat permission
    # if not 'product_semantic_search_beta' in current_user.permission:
    #   raise fastapi.HTTPException(
    #       status_code=fastapi.status.HTTP_403_FORBIDDEN,
    #       detail="User does not have permission to access this resource")

    await websocket.accept()

    while True:
        try:
            # get model type and messages from the client
            data = await websocket.receive_text()
        except Exception as ex:
            # this is a disconnect on the client side
            break

        data = json.loads(data)
        messages = data["messages"]

        # TODO: this has to be changed
        has_metadata = data.get("meta", None) is not None
        if has_metadata:
            message_type = data["meta"].get("command_type", "Unknown")
        else:
            message_type = "Unknown"
        
        prompt_tokens = make_input_tokens(messages=messages, method="ours")
        request_id = random.randint(0, 10_000_000_000)

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
            repeat_retries=3,
        )

        client_hyperparameters = data.get("hyperparameters", None)
        if client_hyperparameters is not None:
            hyperparameters = replace(hyperparameters, **client_hyperparameters)

        client_anti_repeat_policy = data.get("anti_repeat_policy", None)
        if client_anti_repeat_policy is not None:
            anti_repeat_policy = replace(anti_repeat_policy, **client_anti_repeat_policy)

        client_model = data.get("model", None) # TODO unused so far

        server_info = LLMServerInfo(
            url="localhost:8001",
            model_name="tensorrt_llm",
        )

        # print("Decoded: ", tokenizer.decode(prompt_tokens))
        generated_message_chunks = []
        generated_tokens = 0

        token_stream = decode_token_stream(
            request_id,
            prompt_tokens,
            tokenizer,
            server_info,
            hyperparameters,
            anti_repeat_policy
        )

        queue = asyncio.Queue()
        thread = threading.Thread(target=run_generator, args=(token_stream, queue))
        thread.start()

        start_llm_completion_time = time.time()

        # consume from the queue
        while True:
            # TODO timeout?
            # there is a timeout, TODO can be removed
            try:
                item = await queue.get()
                if item is END_OF_GENERATOR:
                    termination_reason = "generation_complete"
                    break
                else:
                    item: CompletionResponse
                    generated_message_chunks.append(item.content)
                    generated_tokens += item.total_tokens
                # send to client
                    await websocket.send_text(
                        json.dumps({"role": "assistant_part", "result": item.as_json()})
                    )
            except TimeoutException:
                termination_reason = "timeout"
                break
            except InferenceServerException:
                termination_reason = "server_error"
                break
            except RetryLimitExceeded:
                termination_reason = "retry_limit_exceeded"
                break
            except Exception as ex:
                termination_reason = "unknown error"
                break

        llm_completion_duration = time.time() - start_llm_completion_time

        completion_message = {
            "role": "server",
            "content": "".join(generated_message_chunks),
            "termination_reason": termination_reason
        }

        transaction = {
            # current_user is a dict
            "user": current_user["id"],
            "input": messages,
            "output": completion_message,
            "command": message_type,
            "timestamp": datetime.datetime.now(),
            "duration": llm_completion_duration,
            "num_completion_tokens": generated_tokens
        }

        print(transaction)

        # Send the final chunk
        await websocket.send_text(
            json.dumps({"role": "server", "content": {"end_of_stream": termination_reason}})
        )

# run the app on localhost port 8085
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Client for the UANKIN LLM WebSocket API')
    parser.add_argument('--visibility', type=str, default="localhost", help='Visibility of the model')
    parser.add_argument('--port', type=int, default=8085, help='Port to run the server on')

    args = parser.parse_args()
    host = "localhost" if args.visibility == "localhost" else "0.0.0.0"
    uvicorn.run(app, host=host, port=args.port)