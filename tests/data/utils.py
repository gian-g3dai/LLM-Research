from termcolor import colored

def prompt_response_print(prompt_tokens, sample_tokens, tokenizer):
    
    # decode prompt and sample tokens
    prompt = tokenizer.decode(prompt_tokens)
    response = tokenizer.decode(sample_tokens)
    
    print("PROMPT")
    print(colored(prompt, "green"))
    print("RESPONSE")
    print(colored(response, "red"))
    print(colored("END"))