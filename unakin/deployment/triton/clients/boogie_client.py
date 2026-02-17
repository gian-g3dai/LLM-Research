import argparse
import json
from dataclasses import asdict, dataclass
from pprint import pprint

import requests
import websocket


@dataclass
class Message:
    role: str
    content: str

@dataclass
class Hyperparameters:
    temperature: float
    max_gen_len: int
    runtime_top_p: float

@dataclass
class AntiRepeatPolicy:
    min_size: int
    max_size: int
    min_repeat_proportion: float
    repeat_retries: int

class CompletionRequest:
    # add checks if exists
    necessary_keys = ["messages", "model", "hyperparameters", "anti_repeat_policy", "meta"]
    
    def __init__(self, request_dict):
        assert isinstance(request_dict, dict), "CompletionRequest must be initialized with a dictionary"
        assert all(key in request_dict for key in self.necessary_keys), f"CompletionRequest must contain all of the following keys: {self.necessary_keys}"
        self.messages = [Message(**msg) for msg in request_dict["messages"]]
        self.model = request_dict["model"]
        self.hyperparameters = Hyperparameters(**request_dict["hyperparameters"])
        self.anti_repeat_policy = AntiRepeatPolicy(**request_dict["anti_repeat_policy"])
        self.meta = request_dict["meta"]
    
    def to_dict(self):
        return {
            "messages": [asdict(msg) for msg in self.messages],
            "model": self.model,
            "hyperparameters": asdict(self.hyperparameters),
            "anti_repeat_policy": asdict(self.anti_repeat_policy),
            "meta": self.meta
        }

class HyperparametersValidator:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def validate_temperature(self):
        assert isinstance(self.hyperparameters.temperature, (int, float)), "Temperature must be a float or an int"
        assert 0.0 <= self.hyperparameters.temperature <= 1.0, "Temperature must be between 0.0 and 1.0"

    def validate_max_gen_len(self):
        assert isinstance(self.hyperparameters.max_gen_len, int), "max_gen_len must be an integer"
        assert self.hyperparameters.max_gen_len > 0, "max_gen_len must be a positive integer"
        assert self.hyperparameters.max_gen_len <= 4096, "max_gen_len must be less than or equal to 4096"

    def validate_runtime_top_p(self):
        assert isinstance(self.hyperparameters.runtime_top_p, (int, float)), "runtime_top_p must be a float or an int"
        assert 0.0 <= self.hyperparameters.runtime_top_p <= 1.0, "runtime_top_p must be between 0.0 and 1.0"

    def validate_all_hyperparameters(self):
        self.validate_temperature()
        self.validate_max_gen_len()
        self.validate_runtime_top_p()

class AntiRepeatPolicyValidator:
    def __init__(self, anti_repeat_policy):
        self.anti_repeat_policy = anti_repeat_policy

    def validate_min_size(self):
        assert isinstance(self.anti_repeat_policy.min_size, int), "min_size must be an integer"
        assert 2 <= self.anti_repeat_policy.min_size <= 16, "min_size must be a positive integer between 2 and 16"
        assert self.anti_repeat_policy.min_size <= self.anti_repeat_policy.max_size, "min_size must be less than or equal to max_size"

    def validate_max_size(self):
        assert isinstance(self.anti_repeat_policy.max_size, int), "max_size must be an integer"
        assert 2 <= self.anti_repeat_policy.max_size <= 16, "max_size must be a positive integer between 2 and 16"

    def validate_min_repeat_proportion(self):
        assert isinstance(self.anti_repeat_policy.min_repeat_proportion, (int, float)), "min_repeat_proportion must be a float or an int"
        assert 0.0 <= self.anti_repeat_policy.min_repeat_proportion <= 1.0, "min_repeat_proportion must be between 0.0 and 1.0"

    def validate_repeat_retries(self):
        assert isinstance(self.anti_repeat_policy.repeat_retries, int), "repeat_retries must be an integer"
        assert self.anti_repeat_policy.repeat_retries >= 0, "repeat_retries must be a non-negative integer"

    def validate_all_anti_repeat_policy(self):
        self.validate_min_size()
        self.validate_max_size()
        self.validate_min_repeat_proportion()
        self.validate_repeat_retries()

class CompletionRequestValidator:
    def __init__(self, completion_request):
        self.completion_request = completion_request

    def validate_messages(self):
        assert isinstance(self.completion_request.messages, list), "messages must be a list"
        assert all(isinstance(msg, Message) for msg in self.completion_request.messages), "Each message must be a Message object"

    def validate_model(self):
        assert isinstance(self.completion_request.model, str), "model must be a string"

    def validate_hyperparameters(self):
        hyperparameters_validator = HyperparametersValidator(self.completion_request.hyperparameters)
        hyperparameters_validator.validate_all_hyperparameters()

    def validate_anti_repeat_policy(self):
        anti_repeat_validator = AntiRepeatPolicyValidator(self.completion_request.anti_repeat_policy)
        anti_repeat_validator.validate_all_anti_repeat_policy()

    def validate_completion_request(self):
        self.validate_messages()
        self.validate_model()
        self.validate_hyperparameters()
        self.validate_anti_repeat_policy()


def iterative_test(ws: websocket.WebSocketApp):
    while True:
        # Receive data from the server
        response = json.loads(ws.recv())

        if response["role"] == "server" and "end_of_stream" in response["content"]:
            print()
            print(response)
            ws.close()
            break
        else:
            print(response['result']['content'], flush=True, end="")


def compact_test(ws: websocket.WebSocketApp):
    full_response = ""
    while True:
        # Receive data from the server
        response = json.loads(ws.recv())

        if response["role"] == "server" and "end_of_stream" in response["content"]:
            # server has finished sending response chunks
            print(response)
            ws.close()
            break
        else:
            full_response += response['result']['content']
    
    print(full_response, flush=True, end="")
    print()
  

def main():
    
    parser = argparse.ArgumentParser(description='Client for the UANKIN LLM WebSocket API')
    parser.add_argument('--server_address', type=str, default="chat.unakin.ai", help='Chat server address')
    parser.add_argument('--user', type=str, required=True, help='Your user name registered on dashboard.unakin.ai')
    parser.add_argument('--password', type=str, required=True, help='Your user password registered on dashboard.unakin.ai')
    parser.add_argument('--compact', action='store_true', help='Compact test')
    args = parser.parse_args()

    # login and get token
    url = f"https://{args.server_address}/auth/token"
    payload = {
        "username": args.user,
        "password": args.password,
    }
    response = requests.post(url, data=payload)

    if response.status_code != 200:
        print(f"Error logging in: {response.status_code} {response.text}")
        exit(1)

    user_token = response.json()["access_token"]

    ws = websocket.create_connection(f"wss://{args.server_address}/ws", header=[f"Authorization: Bearer {user_token}"])

    completion_request = {
        "messages" : [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "The answer to your question is 4."},
            {"role": "user", "content": "What question did I ask?"},
        ],
        # optional:
        "model": "default",
        "hyperparameters": {
            "temperature": 0.8,
            "max_gen_len": 4096,
            "runtime_top_p": 0.9
        },
        "anti_repeat_policy": {
            "min_size": 2,
            "max_size": 8,
            "min_repeat_proportion": 0.5,
            "repeat_retries": 3,
        },
        "meta": {
            "command_type": "chat"
        }
    }


    completion_request = CompletionRequest(completion_request)
    completion_request_validator = CompletionRequestValidator(completion_request)
    completion_request_validator.validate_completion_request()
    completion_request = completion_request.to_dict()

    ws.send(json.dumps(completion_request))    

    if args.compact:
      compact_test(ws)
    else:
      iterative_test(ws)

    

if __name__ == "__main__":
    main()