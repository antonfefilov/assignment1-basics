import os
import mmap
from typing import Iterator, Optional
from multiprocessing import Pool, RLock, shared_memory
from functools import partial
from collections import Counter
import regex as re
from tqdm import tqdm

# Constants
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_SPLIT_TOKEN = "<|endoftext|>"

def _init_tqdm(lock):
    """Initialize tqdm with a shared lock for multiprocessing"""
    tqdm.set_lock(lock)


class MemoryMappedProcessor:
    """
    Memory-efficient processor using memory-mapped files and streaming.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        
    def find_chunk_boundaries_mmap(self, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
        """
        Find chunk boundaries using memory-mapped file for efficient seeking.
        """
        chunk_size = self.file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = self.file_size
        
        with open(self.file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                for bi in range(1, len(chunk_boundaries) - 1):
                    start_pos = chunk_boundaries[bi]
                    # Look for boundary within next 64KB
                    search_end = min(start_pos + 65536, self.file_size)
                    
                    # Find the special token in the search window
                    found_at = mmapped_file.find(split_special_token, start_pos, search_end)
                    if found_at != -1:
                        chunk_boundaries[bi] = found_at
                    else:
                        # If not found, keep original boundary
                        chunk_boundaries[bi] = start_pos
        
        return sorted(set(chunk_boundaries))
    
    def process_chunk_mmap(self, start: int, end: int, split_token: str, chunk_id: int) -> dict[tuple[bytes, ...], int]:
        """
        Process a chunk using memory-mapped file for minimal memory usage.
        """
        chunk_size = end - start
        pre_token_counts = {}
        
        with open(self.file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                with tqdm(
                    total=100,
                    desc=f"Processing chunk {chunk_id} ({start}-{end}) {chunk_size / 1024 / 1024:.2f} MB",
                    position=chunk_id,
                    leave=True,
                    unit="%",
                ) as pbar:
                    
                    # Process in smaller sub-chunks to reduce memory usage
                    sub_chunk_size = 1024 * 1024  # 1MB sub-chunks
                    bytes_processed = 0
                    
                    current_pos = start
                    leftover = b""  # Store incomplete stories from previous sub-chunk
                    
                    while current_pos < end:
                        # Calculate sub-chunk boundaries
                        sub_end = min(current_pos + sub_chunk_size, end)
                        
                        # Read sub-chunk from memory-mapped file
                        sub_chunk_bytes = mmapped_file[current_pos:sub_end]
                        
                        # Handle text boundary issues by looking for complete stories
                        if sub_end < end:  # Not the last sub-chunk
                            # Find last complete story boundary
                            last_boundary = sub_chunk_bytes.rfind(split_token.encode('utf-8'))
                            if last_boundary != -1:
                                # Split at story boundary
                                complete_part = sub_chunk_bytes[:last_boundary]
                                leftover = sub_chunk_bytes[last_boundary:]
                                sub_end = current_pos + last_boundary
                            else:
                                # No story boundary found, process as-is
                                complete_part = sub_chunk_bytes
                                leftover = b""
                        else:
                            # Last sub-chunk, process everything
                            complete_part = leftover + sub_chunk_bytes
                            leftover = b""
                        
                        # Decode and process the complete part
                        if complete_part:
                            try:
                                text = complete_part.decode("utf-8", errors="ignore")
                                self._process_text_chunk(text, split_token, pre_token_counts)
                            except UnicodeDecodeError:
                                # Skip problematic chunks
                                pass
                        
                        # Update progress
                        bytes_processed = sub_end - start
                        progress = int((bytes_processed / chunk_size) * 100)
                        pbar.n = min(progress, 100)
                        pbar.refresh()
                        
                        current_pos = sub_end
                    
                    # Process any remaining leftover
                    if leftover:
                        try:
                            text = leftover.decode("utf-8", errors="ignore")
                            self._process_text_chunk(text, split_token, pre_token_counts)
                        except UnicodeDecodeError:
                            pass
                    
                    pbar.n = 100
                    pbar.set_description(f"Completed chunk {chunk_id} - {len(pre_token_counts)} unique tokens")
                    pbar.refresh()
        
        return pre_token_counts
    
    def _process_text_chunk(self, text: str, split_token: str, pre_token_counts: dict):
        """
        Process a text chunk and update token counts.
        """
        stories = self._split_stories_generator(text, split_token)
        
        for story in stories:
            if not story.strip():
                continue
            
            # Tokenize using regex pattern
            for match in re.finditer(PAT, story):
                word = match.group()
                word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                pre_token_counts[word_bytes] = pre_token_counts.get(word_bytes, 0) + 1
    
    def _split_stories_generator(self, text: str, split_token: str) -> Iterator[str]:
        """
        Generator to split stories without loading all at once.
        """
        pattern = re.escape(split_token)
        start = 0
        for match in re.finditer(pattern, text):
            if match.start() > start:
                yield text[start:match.start()]
            start = match.end()
        # Yield remaining text
        if start < len(text):
            yield text[start:]


def pretokenize_mmap(file_path: str, special_tokens: list[str], num_processes: int, split_token: str = DEFAULT_SPLIT_TOKEN) -> dict[tuple[bytes, ...], int]:
    """
    Memory-efficient pretokenization using memory-mapped files.
    
    Args:
        file_path: Path to the input file
        special_tokens: List of special tokens (unused in current implementation)
        num_processes: Number of processes to use for parallel processing
        split_token: Token used to split the file into chunks
        
    Returns:
        Dictionary mapping byte tuples to their frequencies
    """
    processor = MemoryMappedProcessor(file_path)
    boundaries = processor.find_chunk_boundaries_mmap(num_processes, split_token.encode("utf-8"))
    
    # Create arguments for each chunk
    chunk_args = [
        (start, end, split_token, i, file_path)
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]
    
    # Process chunks in parallel
    lock = RLock()
    with Pool(processes=num_processes, initializer=_init_tqdm, initargs=(lock,)) as pool:
        results = pool.map(process_chunk_mmap_wrapper, chunk_args)
    
    # Combine results efficiently
    frequency_table = Counter()
    for result in results:
        frequency_table.update(result)
    
    print("Calculated frequency table ...")
    return frequency_table


def process_chunk_mmap_wrapper(args: tuple[int, int, str, int, str]) -> dict[tuple[bytes, ...], int]:
    """
    Wrapper function for multiprocessing.
    """
    start, end, split_token, chunk_id, file_path = args
    processor = MemoryMappedProcessor(file_path)
    return processor.process_chunk_mmap(start, end, split_token, chunk_id)


# Streaming BPE data structures for reduced memory usage
class StreamingBPEData:
    """
    Memory-efficient BPE data structures using generators and lazy loading.
    """
    
    def __init__(self, frequency_table: dict[tuple[bytes, ...], int]):
        self.frequency_table = frequency_table
        self._word_cache = None
        self._max_cache_size = 10000  # Limit cached words
        
    def get_words_generator(self):
        """
        Generator for words to avoid loading all into memory at once.
        """
        for word_tuple in self.frequency_table.keys():
            yield list(word_tuple)
    
    def build_initial_structures_streaming(self):
        """
        Build initial BPE structures using streaming approach.
        """
        from collections import defaultdict
        
        pair2pos = defaultdict(set)
        freq_pairs = Counter()
        words = []
        
        # Process words in batches to control memory usage
        batch_size = 1000
        batch = []
        
        for word_id, word_tuple in enumerate(self.frequency_table.keys()):
            word = list(word_tuple)
            words.append(word)
            batch.append((word_id, word, self.frequency_table[word_tuple]))
            
            if len(batch) >= batch_size:
                self._process_word_batch(batch, pair2pos, freq_pairs)
                batch = []
        
        # Process remaining batch
        if batch:
            self._process_word_batch(batch, pair2pos, freq_pairs)
        
        return words, pair2pos, freq_pairs
    
    def _process_word_batch(self, batch, pair2pos, freq_pairs):
        """
        Process a batch of words to update pair structures.
        """
        for word_id, word, frequency in batch:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair2pos[pair].add((word_id, i))
                freq_pairs[pair] += frequency