from collections import Counter, defaultdict
from tqdm import tqdm
import json
import pickle
import gc
from pathlib import Path
from .pretokenization_mmap import pretokenize_mmap, StreamingBPEData


class MemoryEfficientBPE:
    """
    Memory-efficient BPE tokenizer using memory-mapped files and streaming.
    """

    def __init__(self):
        self._vocabulary: dict[int, bytes] | None = None
        self._merges: list[tuple[bytes, bytes]] | None = None

    def train(
        self, *, input_path: str, vocab_size: int = 10000, special_tokens: list[str] = [], num_processes: int = 1
    ) -> bool:
        """
        Train a BPE tokenizer with memory-efficient approach.
        
        Key optimizations:
        - Memory-mapped file reading
        - Streaming data processing
        - Garbage collection between steps
        - Reduced data structure duplication
        """
        print("Starting memory-efficient BPE training...")
        
        # Initialize vocabulary
        vocabulary = {}
        vocabulary.update({i: token.encode("utf-8") for i, token in enumerate(special_tokens)})
        vocabulary.update({len(vocabulary) + i: bytes([i]) for i in range(256)})

        # Step 1: Memory-mapped pretokenization
        print("Step 1: Pretokenization with memory mapping...")
        import time
        start_time = time.time()
        
        frequency_table = pretokenize_mmap(input_path, special_tokens, num_processes)
        
        pretokenization_time = time.time() - start_time
        print(f"Pretokenization completed in {pretokenization_time:.2f} seconds")
        
        # Step 2: Build initial structures with streaming
        print("Step 2: Building BPE structures with streaming...")
        streaming_data = StreamingBPEData(frequency_table)
        words, pair2pos, freq_pairs = streaming_data.build_initial_structures_streaming()
        
        # Force garbage collection after building structures
        del streaming_data
        gc.collect()
        
        # Step 3: Memory-efficient BPE merging
        print("Step 3: Performing BPE merges...")
        merges = []
        initial_vocab_size = len(vocabulary)
        num_merges_needed = vocab_size - initial_vocab_size
        
        pbar = tqdm(total=num_merges_needed, desc="Training BPE", unit="merges")
        merge_start_time = time.time()
        
        merge_count = 0
        while len(vocabulary) < vocab_size:
            if not freq_pairs:
                break
            
            # Find best pair
            max_freq = max(freq_pairs.values())
            best_pairs = [p for p, f in freq_pairs.items() if f == max_freq]
            best_pair = max(best_pairs)
            
            # Perform merge with memory cleanup
            self._memory_efficient_merge(
                best_pair[0], best_pair[1], words, pair2pos, freq_pairs, frequency_table
            )
            
            merges.append(best_pair)
            vocabulary[len(vocabulary)] = best_pair[0] + best_pair[1]
            
            merge_count += 1
            pbar.update(1)
            
            # Periodic garbage collection to prevent memory buildup
            if merge_count % 100 == 0:
                gc.collect()
        
        pbar.close()
        merge_time = time.time() - merge_start_time
        print(f"Merge processing completed in {merge_time:.2f} seconds")
        
        # Final cleanup
        del words, pair2pos, freq_pairs, frequency_table
        gc.collect()
        
        # Store results
        self._vocabulary = vocabulary
        self._merges = merges
        
        print(f"Training completed. Final vocabulary size: {len(vocabulary)}")
        return True

    def _memory_efficient_merge(self, a, b, word_list, pair2pos, freq_pairs, frequency_table):
        """
        Memory-efficient merge operation with better cleanup.
        """
        ab = a + b
        
        # Get positions and sort in reverse order for safe modification
        positions = list(pair2pos[(a, b)])
        positions.sort(reverse=True)
        
        # Track words to update frequency table
        words_to_update = {}
        
        # Process each position
        for word_id, pos in positions:
            if (word_id, pos) not in pair2pos[(a, b)]:
                continue
                
            tokens = word_list[word_id]
            original_tokens = tuple(tokens)
            word_frequency = frequency_table[original_tokens]
            
            # Verify pair exists at position
            if pos >= len(tokens) - 1 or tokens[pos] != a or tokens[pos + 1] != b:
                continue
            
            # Get adjacent tokens
            prev_token = tokens[pos - 1] if pos > 0 else None
            next_token = tokens[pos + 2] if pos < len(tokens) - 2 else None
            
            # Update pair tracking structures
            self._update_pair_structures_for_merge(
                word_id, pos, prev_token, next_token, a, b, ab, 
                word_frequency, pair2pos, freq_pairs
            )
            
            # Perform the actual merge
            tokens[pos:pos + 2] = [ab]
            
            # Add new pairs
            if prev_token is not None:
                pair2pos[(prev_token, ab)].add((word_id, pos - 1))
                freq_pairs[(prev_token, ab)] += word_frequency
                
            if next_token is not None:
                pair2pos[(ab, next_token)].add((word_id, pos))
                freq_pairs[(ab, next_token)] += word_frequency
            
            # Track for frequency table update
            words_to_update[word_id] = (original_tokens, tuple(tokens), word_frequency)
            
            # Update positions for remaining pairs in this word
            self._update_remaining_positions(word_id, pos, tokens, pair2pos)
        
        # Batch update frequency table
        for word_id, (old_tokens, new_tokens, frequency) in words_to_update.items():
            del frequency_table[old_tokens]
            frequency_table[new_tokens] = frequency
        
        # Clean up the merged pair
        pair2pos[(a, b)].clear()
        if (a, b) in pair2pos:
            del pair2pos[(a, b)]
        if (a, b) in freq_pairs:
            del freq_pairs[(a, b)]
    
    def _update_pair_structures_for_merge(self, word_id, pos, prev_token, next_token, 
                                        a, b, ab, word_frequency, pair2pos, freq_pairs):
        """
        Update pair tracking structures when performing a merge.
        """
        # Remove old pairs
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
    
    def _update_remaining_positions(self, word_id, merge_pos, tokens, pair2pos):
        """
        Update positions in pair2pos for all remaining pairs in the word.
        """
        # Update positions that come after the merge position
        for i in range(merge_pos, len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            # Remove old position
            pair2pos[pair].discard((word_id, i + 1))
            # Add new position
            pair2pos[pair].add((word_id, i))

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

    def save(self, file_path: str):
        """Save the BPE tokenizer to a file."""
        if self._vocabulary is None or self._merges is None:
            raise ValueError("Tokenizer must be trained before saving")

        data = {"vocabulary": self._vocabulary, "merges": self._merges}
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def load(self, file_path: str):
        """Load the BPE tokenizer from a file."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            self._vocabulary = data["vocabulary"]
            self._merges = data["merges"]

    def save_vocabulary(self, file_path: str):
        """Save the vocabulary to a JSON file."""
        if self._vocabulary is None:
            raise ValueError("Vocabulary must be trained before saving")

        vocab_serializable = {
            str(token_id): token.decode("latin-1") if isinstance(token, bytes) else token
            for token_id, token in self._vocabulary.items()
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(vocab_serializable, f, indent=2, ensure_ascii=False)

    def load_vocabulary(self, file_path: str):
        """Load the vocabulary from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            vocab_serializable = json.load(f)

        self._vocabulary = {
            int(token_id): token.encode("latin-1") if isinstance(token, str) else token
            for token_id, token in vocab_serializable.items()
        }

    def save_merges(self, file_path: str):
        """Save the merges to a text file."""
        if self._merges is None:
            raise ValueError("Merges must be trained before saving")

        with open(file_path, "w", encoding="utf-8") as f:
            for merge_a, merge_b in self._merges:
                a_str = merge_a.decode("latin-1")
                b_str = merge_b.decode("latin-1")
                f.write(f"{a_str} {b_str}\n")

    def load_merges(self, file_path: str):
        """Load the merges from a text file."""
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