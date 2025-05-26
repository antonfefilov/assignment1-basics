from collections import Counter, defaultdict
from tqdm import tqdm
from .pretokenization import pretokenize


def train(
    *, input_path: str, vocab_size: int = 10000, special_tokens: list[str] = [], num_processes: int = 1
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocabulary = {i: bytes([i]) for i in range(256)}
    vocabulary.update({len(vocabulary): b"<|endoftext|>"})
    vocabulary.update({len(vocabulary) + i: token.encode("utf-8") for i, token in enumerate(special_tokens)})

    # Example frequency table: {
    #     (b'n', b'o'): 300,
    #     (b'h', b'e', b'l', b'l', b'o'): 500,  # Example of a longer byte sequence
    #     (b'w', b'o', b'r', b'l', b'd'): 400   # Another example with multiple bytes
    # }
    frequency_table = pretokenize(input_path, num_processes)

    # Build pair2pos and freq_pairs once
    pair2pos = defaultdict(set)
    freq_pairs = Counter()
    words = list(frequency_table.keys())

    for word_id, word in enumerate(words):
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair2pos[pair].add((word_id, i))
            freq_pairs[pair] += frequency_table[word]

    merges = []

    # Calculate the number of merges needed
    initial_vocab_size = len(vocabulary)
    num_merges_needed = vocab_size - initial_vocab_size
    
    # Create progress bar
    pbar = tqdm(total=num_merges_needed, desc="Training BPE", unit="merges")

    while len(vocabulary) < vocab_size:
        if not freq_pairs:
            break

        max_freq = max(freq_pairs.values())
        best_pairs = [p for p, f in freq_pairs.items() if f == max_freq]
        best_pair = max(best_pairs)

        bpe_merge_inplace(best_pair[0], best_pair[1], words, pair2pos, freq_pairs)

        merges.append(best_pair)
        vocabulary[len(vocabulary)] = bytes(best_pair[0] + best_pair[1])

        # Update progress bar
        pbar.update(1)
    
    # Close progress bar
    pbar.close()

    return vocabulary, merges


# Single merge step function
def bpe_merge_inplace(a, b, word_list, pair2pos, freq_pairs):
    ab = a + b

    # Sort positions in reverse order to avoid index shifting when merging tokens.
    # When merging tokens at position i, all positions < i remain unchanged,
    # but positions > i shift left by 1. Processing from right to left ensures
    # we don't invalidate the positions we haven't processed yet.
    positions = sorted(pair2pos[(a, b)], reverse=True)

    for word_id, pos in positions:
        tokens = word_list[word_id]
        # Relevance
        if pos >= len(tokens) - 1 or tokens[pos] != a or tokens[pos + 1] != b:
            continue

        prev_token = tokens[pos - 1] if pos > 0 else None
        next_token = tokens[pos + 2] if pos < len(tokens) - 2 else None

        # Decrease frequencies and remove old pairs
        if prev_token is not None:
            pair2pos[(prev_token, a)].discard((word_id, pos - 1))
            freq_pairs[(prev_token, a)] -= 1
            if freq_pairs[(prev_token, a)] == 0:
                del freq_pairs[(prev_token, a)]
        pair2pos[(a, b)].discard((word_id, pos))
        freq_pairs[(a, b)] -= 1
        if freq_pairs[(a, b)] == 0:
            del freq_pairs[(a, b)]
        if next_token is not None:
            pair2pos[(b, next_token)].discard((word_id, pos + 1))
            freq_pairs[(b, next_token)] -= 1
            if freq_pairs[(b, next_token)] == 0:
                del freq_pairs[(b, next_token)]

        # Merge
        tokens[pos : pos + 2] = [ab]

        # Add new pairs
        if prev_token is not None:
            pair2pos[(prev_token, ab)].add((word_id, pos - 1))
            freq_pairs[(prev_token, ab)] += 1
        if next_token is not None:
            pair2pos[(ab, next_token)].add((word_id, pos))
            freq_pairs[(ab, next_token)] += 1
