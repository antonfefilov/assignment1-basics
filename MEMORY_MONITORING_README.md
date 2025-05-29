# Memory Monitoring for BPE Training

This document explains how to monitor RAM consumption during BPE tokenizer training. Several monitoring tools and approaches are available, each suited for different use cases.

## Quick Start

For immediate memory monitoring of your BPE training, use the enhanced profiling script:

```bash
# Monitor memory only (no CPU profiling)
python profile_training.py --vocab-size 1000 --memory-only --save-memory-log

# Monitor both CPU and memory
python profile_training.py --vocab-size 1000 --save-memory-log
```

## Available Monitoring Tools

### 1. Enhanced Profiling Script (`profile_training.py`)

**Best for:** Comprehensive profiling with detailed memory statistics

**Features:**
- Real-time memory monitoring during training
- Peak, average, and minimum memory usage
- Memory growth analysis
- Optional CPU profiling
- Save memory logs to CSV files

**Usage:**
```bash
# Memory monitoring only
python profile_training.py --vocab-size 1000 --memory-only

# With memory log saving
python profile_training.py --vocab-size 1000 --memory-only --save-memory-log

# Adjust monitoring interval (default: 0.1s)
python profile_training.py --vocab-size 1000 --memory-only --monitor-interval 0.5

# Both CPU and memory profiling
python profile_training.py --vocab-size 1000 --save-memory-log
```

**Output Example:**
```
Initial memory usage: 22.26 MB
Started memory monitoring (interval: 0.1s)
Training BPE: 100%|████████████████| 743/743 [00:02<00:00, 350.45merges/s]
Training completed in 2.15 seconds
Stopped memory monitoring. Peak memory: 45.67 MB

==================================================
MEMORY USAGE STATISTICS
==================================================
Peak Memory Usage:    45.67 MB
Average Memory Usage: 38.42 MB
Minimum Memory Usage: 22.26 MB
Memory Growth:        +23.41 MB
Total Samples:        22
Monitoring Duration:  2.15 seconds
==================================================
```

### 2. Standalone Memory Monitor (`memory_monitor.py`)

**Best for:** Monitoring external processes or long-running training jobs

**Features:**
- Monitor any process by PID or name
- Real-time memory and CPU usage display
- Process discovery
- CSV logging

**Usage:**
```bash
# List all Python processes
python memory_monitor.py --list-python

# Monitor by process name
python memory_monitor.py --process-name python --save-log training_memory.csv

# Monitor by PID
python memory_monitor.py --pid 12345 --save-log memory.csv

# Monitor for specific duration
python memory_monitor.py --process-name python --duration 300  # 5 minutes

# Adjust monitoring interval
python memory_monitor.py --process-name python --interval 2.0  # Every 2 seconds
```

**Real-time Output:**
```
Starting memory monitoring (interval: 1.0s)
Press Ctrl+C to stop monitoring

Time:   15.2s | Memory:   45.67 MB | Peak:   45.67 MB | CPU:  85.3%
```

### 3. Integrated Monitoring (`BPEWithMonitoring`)

**Best for:** Built-in monitoring in your own code

**Features:**
- Drop-in replacement for the standard BPE class
- Automatic memory tracking
- Programmatic access to memory statistics
- Configurable monitoring intervals

**Usage:**
```python
from tokenizer.bpe_with_monitoring import BPEWithMonitoring

# Create BPE with monitoring
bpe = BPEWithMonitoring(monitor_interval=0.1)

# Train with monitoring enabled
bpe.train(
    input_path="data/training.txt",
    vocab_size=10000,
    special_tokens=["<|pad|>"],
    enable_monitoring=True,
    save_memory_log=True
)

# Get memory statistics
stats = bpe.get_memory_stats()
print(f"Peak memory: {stats['peak_memory_mb']:.2f} MB")
```

### 4. Convenience Function

**Best for:** Quick one-liner training with monitoring

**Usage:**
```python
from tokenizer.bpe_with_monitoring import train_bpe_with_monitoring

# Train and get both BPE and memory stats
bpe, memory_stats = train_bpe_with_monitoring(
    input_path="data/training.txt",
    vocab_size=10000,
    special_tokens=["<|pad|>"],
    save_memory_log=True
)

print(f"Training completed. Peak memory: {memory_stats['peak_memory_mb']:.2f} MB")
```

## Memory Log Format

All tools can save memory usage data to CSV files with the following format:

```csv
timestamp_seconds,memory_mb,cpu_percent
0.000,22.26,0.0
0.100,23.45,15.2
0.200,25.67,32.1
...
```

## Monitoring Different Scenarios

### Large Vocabulary Training
For large vocabulary sizes (>50k), use more frequent monitoring:

```bash
python profile_training.py --vocab-size 50000 --memory-only --monitor-interval 0.05
```

### Long-Running Training
For training that takes hours, use the standalone monitor:

```bash
# Terminal 1: Start training
python -c "from tokenizer.bpe import BPE; bpe = BPE(); bpe.train(input_path='large_dataset.txt', vocab_size=100000)"

# Terminal 2: Monitor the process
python memory_monitor.py --process-name python --save-log long_training.csv
```

### Batch Processing
For monitoring multiple training runs:

```python
from tokenizer.bpe_with_monitoring import train_bpe_with_monitoring

vocab_sizes = [1000, 5000, 10000, 20000]
results = []

for vocab_size in vocab_sizes:
    bpe, stats = train_bpe_with_monitoring(
        input_path="data/training.txt",
        vocab_size=vocab_size,
        save_memory_log=True
    )
    results.append((vocab_size, stats['peak_memory_mb']))
    print(f"Vocab {vocab_size}: {stats['peak_memory_mb']:.2f} MB peak")
```

## Understanding Memory Usage

### Memory Types Monitored
- **RSS (Resident Set Size):** Physical memory currently used by the process
- **Peak Memory:** Maximum memory usage during training
- **Memory Growth:** Difference between peak and initial memory

### Typical Memory Patterns
1. **Initial spike:** Loading and preprocessing data
2. **Gradual growth:** Building frequency tables and merge operations
3. **Peak during merges:** Memory-intensive merge operations
4. **Possible fluctuations:** Garbage collection and data structure updates

### Memory Optimization Tips
- Use smaller monitoring intervals for detailed analysis
- Monitor memory growth to identify memory leaks
- Compare memory usage across different vocabulary sizes
- Use the standalone monitor for long-running processes to avoid overhead

## Troubleshooting

### High Memory Usage
If you see unexpectedly high memory usage:
1. Check the input file size and vocabulary size
2. Monitor memory growth over time
3. Consider using smaller vocabulary sizes for testing
4. Use the detailed profiling to identify memory hotspots

### Monitoring Overhead
The monitoring tools have minimal overhead:
- Enhanced profiling: ~0.1% CPU overhead
- Standalone monitor: No overhead on target process
- Integrated monitoring: ~0.05% CPU overhead

### Process Not Found
If the standalone monitor can't find your process:
```bash
# List all Python processes
python memory_monitor.py --list-python

# Use the exact PID instead of process name
python memory_monitor.py --pid <exact_pid>
```

## Examples

See `example_memory_monitoring.py` for complete working examples of all monitoring approaches.

```bash
# Run all examples
python example_memory_monitoring.py
```

## Dependencies

All monitoring tools use the `psutil` library, which is already included in your project dependencies. No additional installation required. 