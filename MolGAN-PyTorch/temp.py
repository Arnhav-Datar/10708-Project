from transformers import AutoTokenizer, BertModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, love you 3 30 300 3000 30000", return_tensors="pt")

print(inputs.input_ids)
# Decode the inputs ids into token text
print(tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))

