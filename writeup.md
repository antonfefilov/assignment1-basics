# Problem (unicode1): Understanding Unicode (1 point)

(a) What Unicode character does chr(0) return?
Answer: \x00

(b) How does this character’s string representation (__repr__()) differ from its printed representa-
tion?
Answer: __repr__() produces the representation of the string, it wraps the character in quotes and shows
the byte/Unicode escape \x00, i.e. "'\x00'". print() emits the actual character, because \x00
is a control character with no visual glyph, nothing appears on-screen.

(c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
Answer: In string representation it will be displayed as \x00, and will be no symbol when printed.


# Problem (unicode2): Unicode Encodings (3 points)

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
Answer:
 - The sequence become longer => more memory to store the encoded string.
 - Most of public text corpora are already stored as UTF-8 => no need to preprocess
 - Compact, fixed alphabet (256 symbols). UTF-16 has 65 536 possible code-units. UTF-32 has 4 294 967 296.

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
return ""
.join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
Answer: "ち" will break this function. The reason: the function decode input sequence one by one, but UTF-8
doesn't quarantee "one symbol = one byte", the encoding of one symbol might vary from one to four bytes.

(c) Give a two byte sequence that does not decode to any Unicode.
Answer: b'\xe3\x81'. \xe3 the UTF-8 grammar defines as the lead byte of a 3-byte sequence.

# Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

Core compute - RunPod's cpu5c-8-16 instance:
vCPUs: 8
RAM: 16 GiB DDR5
HD: 32Gb NVMe
Processor: AMD EPYC 4564P 16-Core Processor 

(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size
of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary.
Serialize the resulting vocabulary and merges to disk for further inspection. How many hours
and memory did training take? What is the longest token in the vocabulary? Does it make sense?
Resource requirements: ≤30 minutes (no GPUs), ≤30GB RAM
Hint You should be able to get under 2 minutes for BPE training using multiprocessing during
pretokenization and the following two facts:
(a) The <|endoftext|> token delimits documents in the data files.
(b) The <|endoftext|> token is handled as a special case before the BPE merges are applied.

Q: How many hours and memory did training take?.
A: Approx 1 minute, around 1,3Gb. Data corpus is around 2,7Gb.

Q: What is the longest token in the vocabulary?
A: there are 3, " responsibility", " disappointment", " accomplishment" 

(b) Profile your code. What part of the tokenizer training process takes the most time?

Q: What part of the tokenizer training process takes the most time?
A: Pretokenization.

# Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

Core compute - RunPod's cpu5c-8-16 instance:
vCPUs: 8
RAM: 16 GiB DDR5
HD: 32Gb NVMe
Processor: AMD EPYC 4564P 16-Core Processor 

(a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary
size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What
is the longest token in the vocabulary? Does it make sense?
Resource requirements: ≤12 hours (no GPUs), ≤100GB RAM
Deliverable: A one-to-two sentence response.
(b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
Deliverable: A one-to-two sentence response.

Q: How many hours and memory did training take?.
A: Approx 2 hrs 10 minutes, around 12Gb. Data corpus is around 11Gb.

Q: What is the longest token in the vocabulary?
A: there are 2, " disproportionately" and " telecommunications"

Q: Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
A: The main difference is that in OWT tokenizer the most time was spend on merge steps, it's clear that the relationship between the corpus size and training time is not linear. 