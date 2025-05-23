import os
from typing import BinaryIO
from multiprocessing import Pool
from collections import Counter


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize(file_path: str, num_processes: int) -> dict[tuple[bytes, ...], int]:
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

        # Create list of (start, end, file_path) tuples for each chunk
        chunk_args = [(start, end, file_path) for start, end in zip(boundaries[:-1], boundaries[1:])]

        # Process chunks in parallel using a process pool
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_chunk, chunk_args)

    merged = sum((Counter(d) for d in results), Counter())
    frequency_table = dict(merged)

    return frequency_table


def process_chunk(args: tuple[int, int, str]) -> dict[tuple[bytes, ...], int]:
    """
    Processes a chunk of a file to count word occurrences.

    Args:
        args (tuple[int, int, str]): A tuple containing the start and end byte positions
        of the chunk and the file path.

    Returns:
        dict[tuple[bytes, ...], int]: A dictionary where the keys are tuples of bytes
        representing words, and the values are their respective counts.
    """
    start, end, file_path = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # Initialize empty dictionary to store pre-token counts
        pre_token_counts = {}

        # Split chunk by endoftext token and process each story
        stories = chunk.split("<|endoftext|>")

        # Process each story
        for story in stories:
            # Skip empty stories
            if not story.strip():
                continue

            # Split story into words and count occurrences
            words = story.split()
            for word in words:
                word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                pre_token_counts[word_bytes] = pre_token_counts.get(word_bytes, 0) + 1

        return pre_token_counts