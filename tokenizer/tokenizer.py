import json
import ijson
import regex as re
from typing import Iterable, Iterator


class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        Initializes the Tokenizer with vocabulary, merges, and optional special tokens.
        :param vocab: A dictionary mapping token IDs to byte sequences.
        :param merges: A list of tuples representing byte pairs to be merged.
        :param special_tokens: A list of special tokens (optional) to be used by the tokenizer.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        # Initialize special tokens if provided
        if self.special_tokens:
            max_key = max(vocab.keys())  # Find the maximum key in the vocabulary
            for token in self.special_tokens:
                token_utf8 = token.encode("utf-8")
                if token_utf8 not in vocab.values():
                    # Assign a new key for the special token
                    vocab[max_key + 1] = token_utf8
                    max_key += 1

        self.reverse_vocab = {v: k for k, v in vocab.items()}  # Reverse mapping for encoding

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None =None):
        """
        Class method to create a Tokenizer instance from vocabulary and merges files.
        :param vocab_filepath: Path to the vocabulary file.
        :param merges_filepath: Path to the merges file.
        :param special_tokens: Optional list of special tokens.
        :return: An instance of Tokenizer.
        """
        vocab = {}

        with open(vocab_filepath, "r") as f:
            parser = ijson.kvitems(f, "")
            for key, value in parser:
                vocab[int(key)] = value.encode("utf-8")

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = [
                tuple(s.encode() for s in re.split(r" (?=[^ ]+$)", line.strip("\n"), maxsplit=1))
                for line in f.readlines()
            ]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encodes a given text into a list of token IDs.
        :param text: The input text to be encoded.
        :return: A list of token IDs corresponding to the input text.
        """
        if not text: return []

        if self.special_tokens:
            special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
            # Create a regex pattern that matches any of the special tokens
            special_pattern = '|'.join(map(re.escape, special_tokens_sorted))

            parts = re.split(f"({special_pattern})", text)
        else:
            parts = [text]

        tokens = []

        for part in parts:
            if not part:
                continue  # skip empty splits
            if part in self.special_tokens:
                tokens.append(self.reverse_vocab.get(part.encode('utf-8')))
            else:
                for token in re.findall(self.PAT, part):
                    tokens.extend(self._merge(token.encode('utf-8')))

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encodes an iterable of strings into a generator of token IDs.
        :param iterable: An iterable containing strings to be encoded.
        :return: A generator yielding token IDs for each string in the iterable.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into text.
        :param ids: A list of token IDs to be decoded.
        :return: The decoded text as a string.
        """
        if not ids: return ''

        return b''.join(self.vocab.get(id) for id in ids).decode('utf-8', errors='replace')


    def _merge(self, token: tuple[bytes, ...]) -> list[int]:
        pre_token = [bytes([byte]) for byte in token]


        while True:
            if len(pre_token) == 1:
                break

            out = False
            merge_out = False

            pairs = list(zip(pre_token, pre_token[1:]))

            for merge in self.merges:
                # for pair in zip(pre_token, pre_token[1:]):
                for i, pair in enumerate(pairs):
                    if pair == merge:
                        pre_token[i] = b''.join(pair)
                        del pre_token[i+1]
                        out = False
                        merge_out = True
                        break
                    else:
                        out = True

                if merge_out:
                    break

            if out:
                break

        ids = list(self.reverse_vocab.get(item) for item in pre_token)

        return ids
