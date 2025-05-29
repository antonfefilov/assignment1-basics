import os
import gc
import psutil
from typing import BinaryIO, Generator, Optional
from multiprocessing import Pool
from collections import Counter
import regex as re
from tqdm import tqdm

# Constants
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
MINI_CHUNK_SIZE = 4096  # Read ahead by 4k bytes at a time
DEFAULT_SPLIT_TOKEN = "<|endoftext|>"
MAX_MEMORY_USAGE_GB = 2.0  # Maximum memory usage per process in GB
STREAM_BUFFER_SIZE = 1024 * 1024  # 1MB buffer for streaming


def get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / (1024**3)


def calculate_optimal_chunk_size(file_size: int, num_processes: int, max_memory_gb: float = MAX_MEMORY_USAGE_GB) -> int:
    """
    Calculate optimal chunk size based on file size, number of processes, and available memory.

    Args:
        file_size: Size of the file in bytes
        num_processes: Number of processes to use
        max_memory_gb: Maximum memory usage per process in GB

    Returns:
        Optimal chunk size in bytes
    """
    # Conservative estimate: assume text expansion factor of 2x when processing
    max_chunk_size = int((max_memory_gb * 1024**3) / 2)

    # Calculate chunk size based on file size and number of processes
    basic_chunk_size = file_size // num_processes

    # Use the smaller of the two to ensure we don't exceed memory limits
    optimal_chunk_size = min(max_chunk_size, basic_chunk_size)

    # Ensure minimum chunk size of 1MB
    return max(optimal_chunk_size, 1024 * 1024)

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

def pretokenize(
    file_path: str,
    special_tokens: list[str],
    num_processes: int,
    split_token: str = DEFAULT_SPLIT_TOKEN,
    max_memory_gb: float = MAX_MEMORY_USAGE_GB,
) -> dict[tuple[bytes, ...], int]:
    """
    Pretokenize a very large file by splitting it into memory-efficient chunks and processing them in parallel.

    Args:
        file_path: Path to the input file
        special_tokens: List of special tokens (unused in current implementation)
        num_processes: Number of processes to use for parallel processing
        split_token: Token used to split the file into chunks
        max_memory_gb: Maximum memory usage per process in GB

    Returns:
        Dictionary mapping byte tuples to their frequencies
    """
    # Get file size and calculate optimal chunk size
    file_size = os.path.getsize(file_path)
    available_memory = get_available_memory_gb()

    print(f"File size: {file_size / (1024**3):.2f} GB")
    print(f"Available memory: {available_memory:.2f} GB")

    # Adjust max memory per process based on available memory
    adjusted_max_memory = min(max_memory_gb, available_memory / (num_processes * 2))
    optimal_chunk_size = calculate_optimal_chunk_size(file_size, num_processes, adjusted_max_memory)

    print(f"Using {adjusted_max_memory:.2f} GB max memory per process")
    print(f"Optimal chunk size: {optimal_chunk_size / (1024**2):.2f} MB")

    with open(file_path, "rb") as f:
        # Calculate number of chunks based on optimal chunk size
        desired_num_chunks = max(num_processes, file_size // optimal_chunk_size)
        boundaries = find_chunk_boundaries(f, desired_num_chunks, split_token.encode("utf-8"))

        print(f"Processing {len(boundaries) - 1} chunks with {num_processes} processes")

        # Create list of (start, end, file_path, split_token, chunk_id, max_memory_gb) tuples for each chunk
        chunk_args = [
            (start, end, file_path, split_token, i, adjusted_max_memory)
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
        ]

        # Process chunks in parallel using a process pool
        with Pool(processes=num_processes) as pool:
            results = []
            with tqdm(total=len(chunk_args), desc="Processing chunks") as pbar:
                for result in pool.imap(process_chunk_streaming, chunk_args):
                    results.append(result)
                    pbar.update(1)

                    # Force garbage collection to free memory
                    gc.collect()

    # Combine results efficiently
    print("Combining results...")
    frequency_table = Counter()
    for result in tqdm(results, desc="Merging frequency tables"):
        frequency_table.update(result)

    return frequency_table


def stream_stories_from_chunk(
    file_path: str, start: int, end: int, split_token: str, buffer_size: int = STREAM_BUFFER_SIZE
) -> Generator[str, None, None]:
    """
    Stream stories from a file chunk without loading the entire chunk into memory.

    Args:
        file_path: Path to the file
        start: Start byte position
        end: End byte position
        split_token: Token used to split stories
        buffer_size: Size of the read buffer

    Yields:
        Individual stories as strings
    """
    with open(file_path, "rb") as f:
        f.seek(start)
        remaining_bytes = end - start

        buffer = b""
        split_token_bytes = split_token.encode("utf-8")

        while remaining_bytes > 0:
            # Read next chunk
            read_size = min(buffer_size, remaining_bytes)
            chunk = f.read(read_size)
            remaining_bytes -= len(chunk)

            if not chunk:
                break

            buffer += chunk

            # Split buffer by the token
            while split_token_bytes in buffer:
                story_bytes, buffer = buffer.split(split_token_bytes, 1)
                try:
                    story = story_bytes.decode("utf-8", errors="ignore")
                    if story.strip():  # Only yield non-empty stories
                        yield story
                except UnicodeDecodeError:
                    # Skip corrupted data
                    continue

        # Process remaining buffer
        if buffer:
            try:
                story = buffer.decode("utf-8", errors="ignore")
                if story.strip():
                    yield story
            except UnicodeDecodeError:
                pass


def process_chunk_streaming(args: tuple[int, int, str, str, int, float]) -> dict[tuple[bytes, ...], int]:
    """
    Processes a chunk of a file using streaming to minimize memory usage.

    Args:
        args: A tuple containing (start, end, file_path, split_token, chunk_id, max_memory_gb)
            start: Start byte position of the chunk
            end: End byte position of the chunk
            file_path: Path to the file being processed
            split_token: Token used to split stories in the chunk
            chunk_id: Unique identifier for this chunk
            max_memory_gb: Maximum memory usage for this process

    Returns:
        dict[tuple[bytes, ...], int]: A dictionary where the keys are tuples of bytes
        representing words, and the values are their respective counts.
    """
    start, end, file_path, split_token, chunk_id, max_memory_gb = args

    # Initialize empty dictionary to store pre-token counts
    pre_token_counts = {}
    story_count = 0

    # Process stories using streaming to minimize memory usage
    try:
        for story in stream_stories_from_chunk(file_path, start, end, split_token):
            story_count += 1

            # Process story and count word occurrences
            for match in re.finditer(PAT, story):
                word = match.group()
                word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                pre_token_counts[word_bytes] = pre_token_counts.get(word_bytes, 0) + 1

            # Periodic memory check and garbage collection
            if story_count % 1000 == 0:
                # Check memory usage
                process = psutil.Process()
                memory_usage_gb = process.memory_info().rss / (1024**3)

                if memory_usage_gb > max_memory_gb:
                    print(
                        f"Warning: Chunk {chunk_id} using {memory_usage_gb:.2f} GB memory (limit: {max_memory_gb:.2f} GB)"
                    )

                # Force garbage collection
                gc.collect()

    except Exception as e:
        print(f"Error processing chunk {chunk_id}: {e}")
        return {}

    print(f"Chunk {chunk_id} processed {story_count} stories, found {len(pre_token_counts)} unique tokens")
    return pre_token_counts


def process_chunk(args: tuple[int, int, str, str, int]) -> dict[tuple[bytes, ...], int]:
    """
    Original process_chunk function for backward compatibility.
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
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # Initialize empty dictionary to store pre-token counts
        pre_token_counts = {}

        # Split chunk by the specified token and process each story
        stories = chunk.split(split_token)

        # Process each story with progress tracking on separate lines
        for story in tqdm(stories, desc=f"Chunk {chunk_id} ({start}-{end})", position=chunk_id, leave=True):
            # Skip empty stories
            if not story.strip():
                continue

            # Split story into words and count occurrences
            for match in re.finditer(PAT, story):
                word = match.group()
                word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                pre_token_counts[word_bytes] = pre_token_counts.get(word_bytes, 0) + 1

        return pre_token_counts