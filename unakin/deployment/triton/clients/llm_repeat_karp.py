def rabin_karp(sequence, pattern):
  if len(pattern) > len(sequence):
    return []

  prime = 32089  
  base = 256   

  pattern_hash = 0
  text_hash = 0
  base_power = 1

  for i in range(len(pattern) - 1):
    base_power = (base_power * base) % prime

  for i in range(len(pattern)):
    pattern_hash = (base * pattern_hash + pattern[i]) % prime
    text_hash = (base * text_hash + sequence[i]) % prime

  occurrences = []
  for i in range(len(sequence) - len(pattern) + 1):
    if pattern_hash == text_hash and sequence[i:i + len(pattern)] == pattern:
      occurrences.append(i)

    if i < len(sequence) - len(pattern):
      text_hash = (base * (text_hash - sequence[i] * base_power) + sequence[i + len(pattern)]) % prime
      if text_hash < 0:
        text_hash += prime

  return occurrences

def find_consecutive_pattern_occurrences(sequence, pattern):
  indices = rabin_karp(sequence, pattern)
  n = len(pattern)
    
  # Check if occurrences are consecutive
  is_consecutive = all(indices[i] == indices[i - 1] + 1 for i in range(1, len(indices))) if indices else False

  return indices, is_consecutive

def get_unique_subsequences(sequence, min_size, max_size):
  subsequences = set()
  for size in range(min_size, max_size + 1):
    for i in range(len(sequence) - size + 1):
      subsequence = tuple(sequence[i:i + size])
      if subsequence not in subsequences:
        subsequences.add(subsequence)

  # convert to list
  subsequences = list(subsequences)
  for i in range(len(subsequences)):
    subsequences[i] = list(subsequences[i])
  return subsequences

def int_list_has_repeats(int_list, min_size, max_size, min_repeat_proportion=0.5):
  unique_sub_lists = get_unique_subsequences(int_list, min_size, max_size)
  for sub_list in unique_sub_lists:
    occurrences, consecutive = find_consecutive_pattern_occurrences(int_list, sub_list)
    if consecutive & (len(occurrences) > 1):
      repeat_size = len(occurrences) * len(sub_list)
      if repeat_size >= min_repeat_proportion * len(int_list):
        print(f"Found repeat of size {repeat_size} with subsequence {sub_list}")
        return True
  return False

if __name__ == "__main__":
  l = [29871, 306, 29915, 29885, 925, 385, 319, 29902, 29892, 306, 1016, 29915, 29873, 505, 1855, 29899]
  print(int_list_has_repeats(l, 2, 8, 0.5))