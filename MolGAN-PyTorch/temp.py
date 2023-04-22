from transformers import AutoTokenizer, BertModel
import torch
import numpy as np
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# inputs = tokenizer("Hello, love you 3 30 300 3000 30000", return_tensors="pt")

# print(inputs.input_ids)
# # Decode the inputs ids into token text
# print(tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))

a = [(1, [1,2]), (2, [3,4])]
b, c = zip(*a)

print(b, c)
print(c[:,1])

