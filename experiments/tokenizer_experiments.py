import random

SEP  = b"<|endoftext|>"

def a():
    """
    Sample 10 documents from TinyStories and OpenWebText.
    Using your previously-trained TinyStories and OpenWebText tokenizers
    (10K and 32K vocabulary size, respectively), encode these sampled documents
    into integer IDs.
    What is each tokenizer’s compression ratio (bytes/token)?
    """

def _reservoir_sampling(file_path, sample_size=10):
    reservoir = []
    buf = b""
    doc_id = 0
    L = len(SEP)

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)  # 4MB chunks

            if not chunk:
                break

            buf += chunk

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

    return reservoir

