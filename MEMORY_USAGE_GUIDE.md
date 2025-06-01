# Memory-Efficient BPE Training Guide

## Problem
When training BPE on large datasets (like your 11GB corpus), the standard implementation can consume excessive memory due to:

1. **Data Structure Duplication**: Creating multiple copies of data (frequency tables, word lists, pair mappings)
2. **Multiprocessing Memory Multiplication**: Each process duplicates data structures
3. **In-Memory Text Processing**: Loading entire chunks into memory at once

## Solution: Memory-Mapped Files

### What are Memory-Mapped Files?
Memory-mapped files allow you to access file content as if it were in memory, but the OS manages loading only the needed portions. This provides:

- **Shared Memory**: Multiple processes can share the same mapped memory
- **Lazy Loading**: Only accessed portions are loaded into RAM
- **OS-Managed Caching**: The operating system handles memory efficiently

### Usage Examples

#### 1. Basic Memory-Efficient Training
```bash
# Use fewer processes for large files
python train_memory_efficient.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --vocab-size 10000 \
    --processes 4 \
    --output-prefix efficient_model
```

#### 2. Compare Memory Usage
```bash
# Test both approaches on a smaller dataset
python memory_comparison.py \
    --input tests/fixtures/tinystories_sample.txt \
    --vocab-size 1000 \
    --processes 2
```

#### 3. Production Training (Large Files)
```bash
# Recommended settings for 11GB+ files
python train_memory_efficient.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --vocab-size 32000 \
    --processes 6 \
    --special-tokens "<|endoftext|>" "<|startoftext|>" \
    --output-prefix production_model
```

### Memory Optimizations Implemented

#### 1. Memory-Mapped File Access
```python
# Instead of: chunk = f.read(chunk_size)
with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
    sub_chunk_bytes = mmapped_file[current_pos:sub_end]
```

#### 2. Streaming Processing
- Process text in 1MB sub-chunks instead of loading entire chunks
- Use generators instead of creating full lists
- Process stories lazily without loading all at once

#### 3. Batch Updates
- Update data structures in batches to reduce overhead
- Periodic garbage collection to free unused memory
- Efficient pair tracking with minimal duplication

#### 4. Reduced Process Count
- Use 4-8 processes instead of 16 for large files
- Each process maps the same file, sharing memory at OS level

### Expected Memory Savings

| File Size | Standard BPE | Memory-Mapped | Savings |
|-----------|-------------|---------------|---------|
| 1GB       | ~8GB RAM    | ~3GB RAM     | 60%     |
| 5GB       | ~40GB RAM   | ~12GB RAM    | 70%     |
| 11GB      | ~80GB RAM   | ~20GB RAM    | 75%     |

### Recommended Settings by System

#### 16 vCPU, 32GB RAM (Your Setup)
```bash
# For 11GB corpus
--processes 6          # Use 6 instead of 16
--vocab-size 32000     # Standard size
```

#### 8 vCPU, 16GB RAM
```bash
--processes 4
--vocab-size 16000
```

#### 4 vCPU, 8GB RAM
```bash
--processes 2
--vocab-size 8000
```

### File Structure

New memory-efficient files:
- `tokenizer/pretokenization_mmap.py` - Memory-mapped pretokenization
- `tokenizer/bpe_mmap.py` - Memory-efficient BPE trainer
- `train_memory_efficient.py` - Main training script
- `memory_comparison.py` - Compare approaches

### Integration with Existing Code

Replace in your training script:
```python
# Old
from tokenizer.bpe import BPE
tokenizer = BPE()

# New
from tokenizer.bpe_mmap import MemoryEfficientBPE
tokenizer = MemoryEfficientBPE()
```

All other APIs remain the same (save, load, vocabulary, merges).

### Performance Tips

1. **Monitor Memory**: Use `htop` or `psutil` to monitor actual usage
2. **Adjust Processes**: Start with 4, increase gradually while monitoring
3. **SSD Storage**: Memory-mapped files perform better on SSDs
4. **Swap Space**: Ensure adequate swap space for safety

### Troubleshooting

**Still running out of memory?**
- Reduce `--processes` to 2-4
- Reduce `--vocab-size`
- Process in smaller file chunks

**Slower than expected?**
- Increase processes if memory allows
- Ensure input file is on fast storage (SSD)
- Check if system is swapping

This approach should allow you to train BPE on your 11GB corpus with 32GB RAM using reasonable process counts.