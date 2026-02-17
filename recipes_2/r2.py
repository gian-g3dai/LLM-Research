import torch
import numpy as np

def slice_lengths_list(lengths: list[int], slice_start: int, slice_stop: int) -> list[int]:
  element_count = slice_stop - slice_start
  acc = 0

  new_lengths = []
  trim_start = False

  for i, l in enumerate(lengths):
    acc += lengths[i]
    # by adding the curent index I'm over the start value
    if acc > slice_start:
      dif = acc - slice_start
      
      if not trim_start:
        lengths[i] -= dif
        trim_start = True
    
    if acc < slice_stop:
      if lengths[i] > 0:
        new_lengths.append(lengths[i])

    elif acc >= slice_stop:
      lengths[i] -= acc - slice_stop
      if lengths[i] > 0:
        new_lengths.append(lengths[i])
      break

  return new_lengths
      
      
if __name__ == "__main__":
  lengths = [3, 4, 5, 6, 7]
  start = 0
  stop = 8
  print(slice_lengths_list(lengths, start, stop))  # [3, 4, 1]