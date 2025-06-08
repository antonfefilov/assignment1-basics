class Tokenizer:
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

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None =None):
        """
        Class method to create a Tokenizer instance from vocabulary and merges files.
        :param vocab_filepath: Path to the vocabulary file.
        :param merges_filepath: Path to the merges file.
        :param special_tokens: Optional list of special tokens.
        :return: An instance of Tokenizer.
        """
        with open(vocab_filepath, 'rb') as f:
            vocab = {int(line.split()[0]): line.split()[1] for line in f.readlines()}

        with open(merges_filepath, 'rb') as f:
            merges = [tuple(line.split()) for line in f.readlines()]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encodes a given text into a list of token IDs.
        :param text: The input text to be encoded.
        :return: A list of token IDs corresponding to the input text.
        """

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
