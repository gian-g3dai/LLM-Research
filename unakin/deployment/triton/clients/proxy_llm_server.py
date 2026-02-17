import asyncio
import json
import threading
import time
import datetime
from pprint import pprint

import fastapi
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import Depends, FastAPI, WebSocket

from unakin.deployment.triton.clients.triton_llm_client_ import (
    TritonLLMClient,
    StreamItem
)

from unakin.llm.tokenizers import LlamaTokenizer

from auth_connector import AuthServiceConnector
from chat_to_tokens import make_input_tokens
from elastic_logger import ElasticLogger


TOKENIZER_PATH = "/home/ubuntu/poseidon/models/tokenizer/tokenizer.model"
tokenizer = LlamaTokenizer(model_path=TOKENIZER_PATH)

END_OF_GENERATOR = object()
GENERATOR_THREAD_TIMEOUT = object()

app = FastAPI()

# API dependencies

def get_auth_enabled():
    return not args.disable_auth

auth_connector = AuthServiceConnector(
    auth_server_url="https://api.unakin.ai/auth",
    frontend_url="http://dashboard.unakin.ai"
)

triton_client = TritonLLMClient(
    url="localhost:8001",
    model_name="tensorrt_llm",
    tokenizer_eos_id=tokenizer.eos_token,
)

try:
    elastic_logger = ElasticLogger(
        host="localhost",
        port=9200,
        user="elastic",
        password="REMOVED"
    )
except Exception as e:
    print("Elasticsearch connection failed: ", e)
    print("This run will not log any messages!")
    elastic_logger = None

app.include_router(auth_connector, prefix="/auth", tags=["Authentication"])


def run_generator(gen, q: asyncio.Queue):
    def worker():
        for item in gen:
            q.put_nowait(item)  # Non-blocking put operation
        q.put_nowait(END_OF_GENERATOR)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=60)

    if thread.is_alive():
        q.put_nowait(GENERATOR_THREAD_TIMEOUT)    

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, auth_enabled = Depends(get_auth_enabled)):
    print("Got headers:", websocket.headers)
    user_real_ip = websocket.headers.get("x-real-ip", None)
    print("New connection request from: ", user_real_ip)

    current_user = None

    if auth_enabled:
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

    await websocket.accept()
    print("Connection accept!")

    while True:
        try:
            # get model type and messages from the client
            data = await websocket.receive_text()
        except Exception as ex:
            # this is a disconnect on the client side
            break

        data = json.loads(data)
        messages = data["messages"]

        # C# frontend's commands, might be deleted in the future
        has_metadata = data.get("meta", None) is not None
        if has_metadata:
            message_type = data["meta"].get("command_type", "Unknown")
        else:
            message_type = "Unknown"
        
        prompt_tokens = make_input_tokens(messages=messages, tokenizer=tokenizer)
        # log this to detect potential tokenizer issues
        tokenized_input = tokenizer.decode(prompt_tokens)

        generated_message_chunks = []
        generated_tokens = 0

        # start generation

        # DANGER! fragile section below, asyncio context might silently deadlock!
        #    DO NOT CHANGE UNLESS THOROUGHLY TESTED
        #    TODO: exact cause TBD

        streams = triton_client.predict([prompt_tokens])
        generator_stream = streams[0]

        queue = asyncio.Queue()
        thread = threading.Thread(target=run_generator, args=(generator_stream, queue))
        thread.start()

        start_llm_completion_time = time.time()

        # consume from the queue
        while True:
            try:
                item = await queue.get()
                
                if item is END_OF_GENERATOR:
                    termination_reason = "generation_complete"
                    break
                elif item is GENERATOR_THREAD_TIMEOUT:
                    termination_reason = "timeout"
                    break
                else:
                    item: StreamItem
                    generated_message_chunks.append(item.content)
                    generated_tokens += item.total_tokens
                    # send to client
                    await websocket.send_text(
                        json.dumps({"role": "assistant_part", "result": item.as_json()})
                    )

            except Exception as ex:
                print("ERROR! Termination caused by exception:", ex)
                termination_reason = "unknown error"
                break

        llm_completion_duration = time.time() - start_llm_completion_time

        # log results

        completion_message = {
            "role": "server",
            "content": "".join(generated_message_chunks),
            "termination_reason": termination_reason
        }

        transaction = {
            "user_id": current_user["id"] if current_user else None,
            "user_email": current_user["email"] if current_user else None,
            "input": messages,
            "tokenized_input": tokenized_input,
            "output": completion_message,
            "command": message_type,
            "timestamp": datetime.datetime.now(),
            "duration": llm_completion_duration,
            "num_completion_tokens": generated_tokens
        }

        pprint(transaction)

        if elastic_logger is not None:
            elastic_logger.log('llm_chat_stats', transaction=transaction)

        # send the final chunk
        await websocket.send_text(
            json.dumps({"role": "server", "content": {"end_of_stream": "generation_complete"}})
        )

    print("Calling websocket close")
    await websocket.close()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"]
)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Client for the UANKIN LLM WebSocket API')
    parser.add_argument('--expose', action='store_true', help='Run on 0.0.0.0 instead of localhost')
    parser.add_argument('--port', type=int, default=8085, help='Port to run the server on')
    parser.add_argument('--disable-auth', action='store_true', help='Disable authentication')

    args = parser.parse_args()

    host = "0.0.0.0" if args.expose else "localhost"
    
    uvicorn.run(app, host=host, port=args.port)
