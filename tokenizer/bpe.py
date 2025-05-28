from collections import Counter, defaultdict
from tqdm import tqdm
import json
import pickle
from pathlib import Path
from .pretokenization import pretokenize


class BPE:
    """
    A Byte Pair Encoding (BPE) tokenizer.
    """

    def __init__(self):
        self._vocabulary: dict[int, bytes] | None = None
        self._merges: list[tuple[bytes, bytes]] | None = None

    def train(
        self, *, input_path: str, vocab_size: int = 10000, special_tokens: list[str] = [], num_processes: int = 1
    ) -> bool:
        """
        Train a BPE tokenizer on the given input file.

        Args:
            input_path: Path to the input training file
            vocab_size: Target vocabulary size (default: 10000)
            special_tokens: List of special tokens to include in vocabulary
            num_processes: Number of processes for parallel pretokenization

        Returns:
            Tuple of (vocabulary dict mapping token_id -> bytes, list of merge operations)
        """
        vocabulary = {}
        # Add special tokens first
        vocabulary.update({i: token.encode("utf-8") for i, token in enumerate(special_tokens)})
        # Add byte tokens
        vocabulary.update({len(vocabulary) + i: bytes([i]) for i in range(256)})

        # Example frequency table: {
        #     (b'n', b'o'): 300,
        #     (b'h', b'e', b'l', b'l', b'o'): 500,  # Example of a longer byte sequence
        #     (b'w', b'o', b'r', b'l', b'd'): 400   # Another example with multiple bytes
        # }
        frequency_table = pretokenize(input_path, special_tokens, num_processes)

        # Build pair2pos and freq_pairs once
        pair2pos = defaultdict(set)
        freq_pairs = Counter()

        words = list(list(key) for key in frequency_table.keys())

        for word_id, word in enumerate(words):
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair2pos[pair].add((word_id, i))
                freq_pairs[pair] += frequency_table[tuple(word)]

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

            self._bpe_merge_inplace(best_pair[0], best_pair[1], words, pair2pos, freq_pairs, frequency_table)

            merges.append(best_pair)
            vocabulary[len(vocabulary)] = best_pair[0] + best_pair[1]

            # Update progress bar
            pbar.update(1)

        # Close progress bar
        pbar.close()

        # Store results in instance variables
        self._vocabulary = vocabulary
        self._merges = merges

        return True

    @property
    def vocabulary(self) -> dict[int, bytes]:
        """Get the trained vocabulary."""
        assert self._vocabulary is not None, "Vocabulary not trained"
        return self._vocabulary

    @property
    def merges(self) -> list[tuple[bytes, bytes]]:
        """Get the trained merge operations."""
        assert self._merges is not None, "Merges not trained"
        return self._merges

    def _bpe_merge_inplace(self, a, b, word_list, pair2pos, freq_pairs, frequency_table):
        """
        Merge all occurrences of the pair (a, b) in the word list in-place.

        Args:
            a: First token of the pair to merge
            b: Second token of the pair to merge
            word_list: List of words (each word is a list of tokens)
            pair2pos: Dictionary mapping pairs to sets of (word_id, position) tuples
            freq_pairs: Counter tracking frequency of each pair
            frequency_table: Dictionary mapping word tuples to their frequencies
        """
        ab = a + b

        # Get all positions for this pair
        positions = list(pair2pos[(a, b)])
        positions.sort(reverse=True)

        # Process each word separately
        for word_id, pos in positions:
            # corner case when the merge change the same pair in the same word
            if (word_id, pos) not in pair2pos[(a, b)]:
                continue

            tokens = word_list[word_id]
            original_tokens = tuple(tokens)
            word_frequency = frequency_table[original_tokens]

            # Verify the pair still exists at this position and fix the position if it doesn't
            if pos >= len(tokens) - 1 or tokens[pos] != a or tokens[pos + 1] != b:
                # Find and update the correct position of the pair
                raise ValueError(
                    f"Pair ({a}, {b}) not found at position {pos} in word number {word_id}. "
                    f"Word tokens: {tokens}, Word length: {len(tokens)}, "
                    f"Expected tokens at pos {pos}: ({tokens[pos] if pos < len(tokens) else 'OUT_OF_BOUNDS'}, "
                    f"{tokens[pos + 1] if pos + 1 < len(tokens) else 'OUT_OF_BOUNDS'}), "
                    f"Actual pair to merge: ({a}, {b})"
                )

            prev_token = tokens[pos - 1] if pos > 0 else None
            next_token = tokens[pos + 2] if pos < len(tokens) - 2 else None

            # Remove old pairs from tracking structures
            if prev_token is not None:
                pair2pos[(prev_token, a)].discard((word_id, pos - 1))
                if len(pair2pos[(prev_token, a)]) == 0:
                    del pair2pos[(prev_token, a)]
                freq_pairs[(prev_token, a)] -= word_frequency
                if freq_pairs[(prev_token, a)] == 0:
                    del freq_pairs[(prev_token, a)]

            if next_token is not None:
                pair2pos[(b, next_token)].discard((word_id, pos + 1))
                if len(pair2pos[(b, next_token)]) == 0:
                    del pair2pos[(b, next_token)]
                freq_pairs[(b, next_token)] -= word_frequency
                if freq_pairs[(b, next_token)] == 0:
                    del freq_pairs[(b, next_token)]

            # Perform the merge
            tokens[pos : pos + 2] = [ab]

            # Add new pairs
            if prev_token is not None:
                pair2pos[(prev_token, ab)].add((word_id, pos - 1))
                freq_pairs[(prev_token, ab)] += word_frequency

            if next_token is not None:
                pair2pos[(ab, next_token)].add((word_id, pos))
                freq_pairs[(ab, next_token)] += word_frequency

            # Update frequency table
            del frequency_table[original_tokens]
            frequency_table[tuple(tokens)] = word_frequency

            # Update positions in pair2pos for all tokens in the word
            for i in range(pos, len(tokens) - 1):
                pair2pos[(tokens[i], tokens[i + 1])].discard((word_id, i + 1))
                pair2pos[(tokens[i], tokens[i + 1])].add((word_id, i))

        # Remove the merged pair from tracking
        pair2pos[(a, b)].clear()
        if (a, b) in pair2pos:
            del pair2pos[(a, b)]
        if (a, b) in freq_pairs:
            del freq_pairs[(a, b)]

    def save(self, file_path: str):
        """
        Save the BPE tokenizer to a file.

        Args:
            file_path: Path to the file where the tokenizer will be saved
        """
        if self._vocabulary is None or self._merges is None:
            raise ValueError("Tokenizer must be trained before saving")

        data = {"vocabulary": self._vocabulary, "merges": self._merges}

        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def load(self, file_path: str):
        """
        Load the BPE tokenizer from a file.

        Args:
            file_path: Path to the file where the tokenizer is saved
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            self._vocabulary = data["vocabulary"]
            self._merges = data["merges"]

    def save_vocabulary(self, file_path: str):
        """
        Save the vocabulary to a JSON file.

        Args:
            file_path: Path to the vocabulary file
        """
        if self._vocabulary is None:
            raise ValueError("Vocabulary must be trained before saving")

        # Convert bytes to base64 strings for JSON serialization
        vocab_serializable = {
            str(token_id): token.decode("latin-1") if isinstance(token, bytes) else token
            for token_id, token in self._vocabulary.items()
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(vocab_serializable, f, indent=2, ensure_ascii=False)

    def load_vocabulary(self, file_path: str):
        """
        Load the vocabulary from a JSON file.

        Args:
            file_path: Path to the vocabulary file
        """
        with open(file_path, "r", encoding="utf-8") as f:
            vocab_serializable = json.load(f)

        # Convert back to the expected format
        self._vocabulary = {
            int(token_id): token.encode("latin-1") if isinstance(token, str) else token
            for token_id, token in vocab_serializable.items()
        }

    def save_merges(self, file_path: str):
        """
        Save the merges to a text file in a human-readable format.

        Args:
            file_path: Path to the merges file
        """
        if self._merges is None:
            raise ValueError("Merges must be trained before saving")

        with open(file_path, "w", encoding="utf-8") as f:
            for merge_a, merge_b in self._merges:
                # Convert bytes to escaped string representation
                a_str = merge_a.decode("latin-1")
                b_str = merge_b.decode("latin-1")
                f.write(f"{a_str} {b_str}\n")

    def load_merges(self, file_path: str):
        """
        Load the merges from a text file.

        Args:
            file_path: Path to the merges file
        """
        merges = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        a_str, b_str = parts
                        merge_a = a_str.encode("latin-1")
                        merge_b = b_str.encode("latin-1")
                        merges.append((merge_a, merge_b))

        self._merges = merges
