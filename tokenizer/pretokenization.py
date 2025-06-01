import os
from typing import BinaryIO
from multiprocessing import Pool, RLock
from functools import partial
from collections import Counter
import regex as re
from tqdm import tqdm

# Constants
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
MINI_CHUNK_SIZE = 4096  # Read ahead by 4k bytes at a time
DEFAULT_SPLIT_TOKEN = "<|endoftext|>"

def _init_tqdm(lock):
    """Initialize tqdm with a shared lock for multiprocessing"""
    tqdm.set_lock(lock)


def split_stories_lazy(text, split_token):
    pattern = re.escape(split_token)
    start = 0
    for match in re.finditer(pattern, text):
        if match.start() > start:
            yield text[start : match.start()]
        start = match.end()
    # Yield remaining text after last split
    if start < len(text):
        yield text[start:]


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

    mini_chunk_size = MINI_CHUNK_SIZE

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

def pretokenize(file_path: str, special_tokens: list[str], num_processes: int, split_token: str = DEFAULT_SPLIT_TOKEN) -> dict[tuple[bytes, ...], int]:
    """
    Pretokenize a file by splitting it into chunks and processing them in parallel.
    
    Args:
        file_path: Path to the input file
        special_tokens: List of special tokens (unused in current implementation)
        num_processes: Number of processes to use for parallel processing
        split_token: Token used to split the file into chunks
        
    Returns:
        Dictionary mapping byte tuples to their frequencies
    """
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token.encode("utf-8"))

        # Create list of (start, end, file_path, split_token, chunk_id) tuples for each chunk
        chunk_args = [
            (start, end, file_path, split_token, i)
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
        ]

        # Process chunks in parallel using a process pool with shared tqdm lock
        lock = RLock()
        with Pool(processes=num_processes, initializer=_init_tqdm, initargs=(lock,)) as pool:
            results = pool.map(process_chunk, chunk_args)

    frequency_table = sum((Counter(d) for d in results), Counter())
    print("Calculated frequency table ...")

    return frequency_table


def process_chunk(args: tuple[int, int, str, str, int]) -> dict[tuple[bytes, ...], int]:
    """
    Processes a chunk of a file to count word occurrences.

    Args:
        args: A tuple containing (start, end, file_path, split_token, chunk_id)
            start: Start byte position of the chunk
            end: End byte position of the chunk
            file_path: Path to the file being processed
            split_token: Token used to split stories in the chunk
            chunk_id: Unique identifier for this chunk

    Returns:
        dict[tuple[bytes, ...], int]: A dictionary where the keys are tuples of bytes
        representing words, and the values are their respective counts.
    """
    start, end, file_path, split_token, chunk_id = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_size = end - start

        # Initialize progress bar for the entire chunk processing
        with tqdm(
            total=100,
            desc=f"Processing chunk {chunk_id} ({start}-{end}) {chunk_size / 1024 / 1024:.2f} MB",
            position=chunk_id,
            leave=True,
            unit="%",
        ) as pbar:
            # Step 1: Load chunk
            pbar.set_description(f"Loading chunk {chunk_id} ({chunk_size / 1024 / 1024:.2f} MB)")
            chunk = f.read(chunk_size)
            chunk = chunk.decode("utf-8", errors="ignore")
            pbar.update(20)  # 20% complete after loading

            # Step 2: Split stories
            pbar.set_description(f"Splitting stories in chunk {chunk_id}")
            stories = split_stories_lazy(chunk, split_token)
            pbar.update(20)  # 40% complete after splitting

            # Initialize empty dictionary to store pre-token counts
            pre_token_counts = {}

            # Step 3: Process stories
            pbar.set_description(f"Processing stories in chunk {chunk_id}")
            for i, story in enumerate(stories):
                # Skip empty stories
                if not story.strip():
                    continue

                # Split story into words and count occurrences
                for match in re.finditer(PAT, story):
                    word = match.group()
                    word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                    pre_token_counts[word_bytes] = pre_token_counts.get(word_bytes, 0) + 1

                # Update progress based on stories processed
                if i % max(1, 100) == 0:  # Update display every 100 stories
                    # Calculate stories per second
                    elapsed_time = pbar.format_dict.get("elapsed", 1)
                    stories_per_sec = i / elapsed_time if elapsed_time > 0 else 0
                    pbar.set_description(
                        f"Chunk {chunk_id}: {i} stories processed ({stories_per_sec:.1f} stories/s) ...",
                        refresh=True,
                    )
                    pbar.refresh()

            # Final update
            pbar.n = 100
            pbar.set_description(f"Completed chunk {chunk_id} - {len(pre_token_counts)} unique tokens")
            pbar.refresh()

        return pre_token_counts