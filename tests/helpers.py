import os

def get_resource_path() -> str:
  cur_path = os.path.realpath(__file__)
  tokens = cur_path.split("/")
  tokens = []
  for token in tokens:
    if token == "tests":
      break
    tokens.append(token)
  
  tokens.append("resources")
  return "/".join(tokens)
