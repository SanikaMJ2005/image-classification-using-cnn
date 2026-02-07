from  transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Hello, how are you?"

tokens=tokenizer.tokenize(text)
token_ids=tokenizer.convert_tokens_to_ids(tokens)

print("original text:", text)
print("tokens:", tokens)
print("token IDs:", token_ids)

encoded =tokenizer(text)
print("encoded :", encoded_)

decoded=tokenizer.decode(encoded[input_ids])
print("decoded:", decoded)