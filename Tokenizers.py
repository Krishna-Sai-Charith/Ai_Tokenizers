# tokenizers_demo.py
"""
Tokenizers Demo – LLaMA 3.1, Phi-3, Qwen2, and Starcoder2
--------------------------------------------------------
This script demonstrates:
1. Loading tokenizers from Hugging Face for different models.
2. Encoding/decoding text and code.
3. Applying chat templates for instruct models.
"""

from huggingface_hub import login
from transformers import AutoTokenizer

# ========== CONFIG ==========
HF_TOKEN = "your_huggingface_token_here"  # Replace with your HF token
LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B"
LLAMA_INSTRUCT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI3_MODEL = "microsoft/Phi-3-mini-4k-instruct"
QWEN2_MODEL = "Qwen/Qwen2-7B-Instruct"
STARCODER2_MODEL = "bigcode/starcoder2-3b"
# ============================

# Login to Hugging Face
login(HF_TOKEN, add_to_git_credential=True)

# ----- LLAMA 3.1 Tokenizer -----
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL, trust_remote_code=True)
text = "I am excited to show Tokenizers in action to my LLM engineers"

tokens = tokenizer.encode(text)
print(f"LLaMA Tokens: {tokens}")
print(f"Number of tokens: {len(tokens)}")
print(f"Decoded: {tokenizer.decode(tokens)}")
print(f"Batch Decoded: {tokenizer.batch_decode(tokens)}")
print("Extra vocab:", tokenizer.get_added_vocab())

# ----- LLAMA 3.1 Instruct Chat Template -----
tokenizer_instruct = AutoTokenizer.from_pretrained(LLAMA_INSTRUCT_MODEL, trust_remote_code=True)
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
]
prompt = tokenizer_instruct.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("\nLLaMA Instruct Prompt:\n", prompt)

# ----- Phi-3 -----
phi3_tokenizer = AutoTokenizer.from_pretrained(PHI3_MODEL)
phi3_tokens = phi3_tokenizer.encode(text)
print("\nPhi-3 Tokens:", phi3_tokens)
print("Phi-3 Batch Decoded:", phi3_tokenizer.batch_decode(phi3_tokens))
print("Phi-3 Chat Template:\n", phi3_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

# ----- Qwen2 -----
qwen2_tokenizer = AutoTokenizer.from_pretrained(QWEN2_MODEL)
qwen2_tokens = qwen2_tokenizer.encode(text)
print("\nQwen2 Tokens:", qwen2_tokens)
print("Qwen2 Chat Template:\n", qwen2_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

# ----- Starcoder2 -----
starcoder2_tokenizer = AutoTokenizer.from_pretrained(STARCODER2_MODEL, trust_remote_code=True)
code_snippet = """
def hello_world(person):
    print("Hello", person)
"""
code_tokens = starcoder2_tokenizer.encode(code_snippet)
print("\nStarcoder2 Code Tokens and Decoding:")
for token in code_tokens:
    print(f"{token} = {starcoder2_tokenizer.decode(token)}")
