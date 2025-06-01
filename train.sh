#!/bin/bash

# BPE Tokenizer Training Script
# This script runs the Python BPE training with configurable parameters

set -e  # Exit on any error

# Default values
INPUT_FILE="data/TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE=10000
SPECIAL_TOKENS='["<|endoftext|>"]'
NUM_PROCESSES=4
MEMORY_INTERVAL=0.5
PYTHON_CMD="python"

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Train a BPE tokenizer with configurable parameters.

OPTIONS:
    -i, --input FILE        Input training file (default: $INPUT_FILE)
    -v, --vocab-size SIZE   Target vocabulary size (default: $VOCAB_SIZE)
    -s, --special-tokens    Special tokens as JSON array (default: $SPECIAL_TOKENS)
    -p, --processes NUM     Number of processes for parallel processing (default: $NUM_PROCESSES)
    -m, --memory-interval   Memory monitoring interval in seconds (default: $MEMORY_INTERVAL)
    --python CMD            Python command to use (default: $PYTHON_CMD)
    -h, --help              Show this help message

EXAMPLES:
    # Basic usage with defaults
    $0

    # Custom vocabulary size and input file
    $0 -i data/my_text.txt -v 5000

    # Multiple special tokens
    $0 -s '["<|endoftext|>", "<|startoftext|>", "<|pad|>"]'

    # Use more processes for faster training
    $0 -p 8

    # Use specific Python version
    $0 --python python3.11

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -v|--vocab-size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        -s|--special-tokens)
            SPECIAL_TOKENS="$2"
            shift 2
            ;;
        -p|--processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        -m|--memory-interval)
            MEMORY_INTERVAL="$2"
            shift 2
            ;;
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file '$INPUT_FILE' does not exist."
    exit 1
fi

if ! [[ "$VOCAB_SIZE" =~ ^[0-9]+$ ]] || [[ "$VOCAB_SIZE" -lt 256 ]]; then
    echo "Error: Vocabulary size must be a positive integer >= 256."
    exit 1
fi

if ! [[ "$NUM_PROCESSES" =~ ^[0-9]+$ ]] || [[ "$NUM_PROCESSES" -lt 1 ]]; then
    echo "Error: Number of processes must be a positive integer."
    exit 1
fi

# Check if Python command exists
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "Error: Python command '$PYTHON_CMD' not found."
    exit 1
fi

# Print configuration
echo "=========================================="
echo "BPE Tokenizer Training Configuration"
echo "=========================================="
echo "Input file:        $INPUT_FILE"
echo "Vocabulary size:   $VOCAB_SIZE"
echo "Special tokens:    $SPECIAL_TOKENS"
echo "Processes:         $NUM_PROCESSES"
echo "Memory interval:   ${MEMORY_INTERVAL}s"
echo "Python command:    $PYTHON_CMD"
echo "=========================================="
echo

# Build the command arguments
PYTHON_ARGS=()
PYTHON_ARGS+=("--input" "$INPUT_FILE")
PYTHON_ARGS+=("--vocab-size" "$VOCAB_SIZE")
PYTHON_ARGS+=("--processes" "$NUM_PROCESSES")
PYTHON_ARGS+=("--memory-interval" "$MEMORY_INTERVAL")

# Parse special tokens JSON array and convert to individual arguments
# Remove brackets and quotes, split by comma
TOKENS=$(echo "$SPECIAL_TOKENS" | sed 's/\[//g' | sed 's/\]//g' | sed 's/"//g' | tr ',' '\n')
if [[ -n "$TOKENS" ]]; then
    PYTHON_ARGS+=("--special-tokens")
    while IFS= read -r token; do
        # Trim whitespace
        token=$(echo "$token" | xargs)
        if [[ -n "$token" ]]; then
            PYTHON_ARGS+=("$token")
        fi
    done <<< "$TOKENS"
fi

# Run the training
echo "Starting BPE training..."
echo "Press Ctrl+C to interrupt if needed."
echo

# Run the Python script with arguments
"$PYTHON_CMD" train.py "${PYTHON_ARGS[@]}"

echo
echo "Training completed successfully!"
echo "Output files have been saved with the base name from the input file." 