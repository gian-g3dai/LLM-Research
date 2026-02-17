import codecs
import subprocess

from typing import List
from enum import Enum

def backtick_code(code: str, language: str) -> str:
  # check if last character in the file is a newline with either \r\n or \n
  if code[-1] == '\n' :
    code = f"```{language}\n{code}```"
  else:
    code = f"```{language}\n{code}\n```"
  return code

def normalize_code(code: str) -> str:
  code = code.strip()
  return code.replace('\r\n', '\n').replace('\r', '\n')

# move those to the code utils file
class GlobbingMethod(Enum):
  all = 0
  user_only = 1
  libs_only = 2

class ReformatMethod(Enum):
  no_reformat = 0
  reformat = 1
  strip_errors_only = 2

def read_file_lines_multiencoding(input_file: str) -> List[str]:
  try:
    with codecs.open(input_file, 'r', encoding='utf-8-sig') as f:
      content = f.readlines()
  except UnicodeDecodeError:
    # If reading with UTF-8 fails, try other encodings
    encodings_to_try = ['utf-16', 'latin-1']
    for encoding in encodings_to_try:
      try:
          with codecs.open(input_file, 'r', encoding=encoding) as f:
            content = f.readlines()
            break
      except UnicodeDecodeError:
          pass
      except UnicodeError:
          pass
    else:
      # If all encodings fail, raise an error
      raise Exception(f"Unable to decode {input_file} with any encoding")

  return content

def _reformat_using_dotnet_formatter(code: str) -> str:
    # Run csharpier to reformat the code
    try:
      # Start the dotnet-format process
      process = subprocess.Popen(
        ["dotnet", "csharpier"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
      )

      # Communicate the code to the process and get the formatted output
      reformatted_code, error_output = process.communicate(input=code)

      # Check if there was any error
      if process.returncode != 0:
        # TODO log which files contain parsing errors and perhaps exclude them from dataset?
        # e.g.: 'else' cannot start a statement. Syntax error, '(' expected ; expected etc.
        pass
        # raise Exception(f"Error occurred while formatting: {error_output}")
      else:
        return reformatted_code
    except Exception as e:
      print("Error:", e)
      raise e