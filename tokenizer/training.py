from collections import Counter
from tqdm import tqdm
from .pretokenization import pretokenize


def train(
    *, input_path: str, vocab_size: int = 10000, special_tokens: list[str] = [], num_processes: int = 1
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocabulary = {i: bytes([i]) for i in range(256)}
    vocabulary.update({len(vocabulary): b"<|endoftext|>"})
    vocabulary.update({len(vocabulary) + i: token.encode("utf-8") for i, token in enumerate(special_tokens)})

    frequency_table = pretokenize(input_path, num_processes)

    merges = []

    # Calculate the number of merges needed
    initial_vocab_size = len(vocabulary)
    num_merges_needed = vocab_size - initial_vocab_size
    
    # Create progress bar
    pbar = tqdm(total=num_merges_needed, desc="Training BPE", unit="merges")

    while len(vocabulary) < vocab_size:
        pf = pairs_freq(frequency_table)
        if not pf:  # Check if pf is empty
            break
        best_freq = max(pf.values())
        # candidates-leaders
        best = max([p for p, f in pf.items() if f == best_freq])  # lexicographic max

        merges.append(best)
        vocabulary[len(vocabulary)] = bytes(best[0] + best[1])

        frequency_table = merge_pair(frequency_table, best)
        
        # Update progress bar
        pbar.update(1)
    
    # Close progress bar
    pbar.close()

    return vocabulary, merges


def pairs_freq(words: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """Frequencies of pairs in {tuple(chars): freq}"""
    freq: Counter = Counter()
    for word, f in words.items():
        for i in range(len(word) - 1):
            freq[(word[i], word[i + 1])] += f
    return freq


def merge_pair(words, pair):
    """Merge pair=(a,b) in all words"""
    a, b = pair
    new = {}
    for word, f in words.items():
        tmp, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                tmp.append(a + b)  # new "symbol"
                i += 2
            else:
                tmp.append(word[i])
                i += 1
        new[tuple(tmp)] = new.get(tuple(tmp), 0) + f
    return new
