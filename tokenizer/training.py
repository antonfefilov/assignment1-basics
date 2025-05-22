from .pretokenization import find_chunk_boundaries
from multiprocessing import Pool
from collections import Counter


def train(file_path: str, num_processes: int) -> list[str]:
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


def process_chunk(args):
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
                pre_token_counts[word] = pre_token_counts.get(word, 0) + 1

        return pre_token_counts
