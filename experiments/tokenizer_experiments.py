from tokenizer.tokenizer import Tokenizer

import random
import time

SEP  = b"<|endoftext|>"
NUMBER_OF_SAMPLES = 10

def a():
    """
    Sample 10 documents from TinyStories and OpenWebText.
    Using your previously-trained TinyStories and OpenWebText tokenizers
    (10K and 32K vocabulary size, respectively), encode these sampled documents
    into integer IDs.
    What is each tokenizer’s compression ratio (bytes/token)?
    """
    ts_samples = _reservoir_sampling("data/TinyStoriesV2-GPT4-valid.txt", NUMBER_OF_SAMPLES)
    # owt_samples = _reservoir_sampling("data/owt_valid.txt", NUMBER_OF_SAMPLES)

    ts_tokenizer = Tokenizer.from_files(vocab_filepath="trainings/tiny_stories_vocabulary.json", merges_filepath="trainings/tiny_stories_merges.txt")
    # owt_tokenizer = Tokenizer.from_files(vocab_filepath="trainings/owt_train_vocabulary.json", merges_filepath="trainings/owt_train_merges.txt")

    # ts_ratios = []
    # for doc in owt_samples:
    #     tokens = ts_tokenizer.encode(doc.decode("utf-8"))
    #     ratio = len(doc) / len(tokens)
    #     ts_ratios.append(ratio)
    #
    # ts_ratio = sum(ts_ratios)/NUMBER_OF_SAMPLES

    total_bytes = sum(len(doc) for doc in ts_samples)

    start_time = time.time()
    for doc in ts_samples:
        tokens = ts_tokenizer.encode(doc.decode("utf-8"))
    end_time = time.time()

    throughput = total_bytes / (end_time - start_time)

    return throughput

def _reservoir_sampling(file_path, sample_size=10):
    import os

    reservoir = []
    buf = b""
    doc_id = 0
    L = len(SEP)

    # Get file size for progress tracking
    file_size = os.path.getsize(file_path)
    bytes_processed = 0

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)  # 40MB chunks

            if not chunk:
                break

            buf += chunk
            bytes_processed += len(chunk)

            # Print progress
            progress_percent = (bytes_processed / file_size) * 100
            print(f"\rProcessed {progress_percent:.1f}% ({bytes_processed:,} / {file_size:,} bytes)", end="", flush=True)

            while True:
                i = buf.find(SEP)

                if i == -1:
                    break

                doc = buf[:i]

                if doc_id < sample_size:
                    reservoir.append(doc)
                else:
                    j = random.randint(0, doc_id)
                    if j < sample_size:
                        reservoir[j] = doc

                doc_id += 1
                buf = buf[i + L:]

    # Handle final doc if file doesn’t end with SEP
    if buf:
        if doc_id < sample_size:
            reservoir.append(buf)
        else:
            j = random.randint(0, doc_id)
            if j < sample_size:
                reservoir[j] = buf
        doc_id += 1

    print()  # New line after progress display
    return reservoir

