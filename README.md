# Tokenizers Demo – LLaMA 3.1, Phi-3, Qwen2, Starcoder2

## 📌 Overview
This project demonstrates how different Hugging Face tokenizers work across various LLM families.  
It focuses on:
- Tokenizing and detokenizing text
- Viewing token IDs and vocabulary differences
- Applying **chat templates** for instruction-tuned models
- Tokenizing code with models specialized for programming languages

---

## 🧠 Key Points
- **Tokenization is model-specific** – the same text will be split differently across models like LLaMA, Phi-3, and Qwen2.
- **Chat models** (like Instruct variants) expect a special prompt structure; `apply_chat_template` handles this automatically.
- **Batch decoding** can be used to convert multiple token IDs back into text efficiently.
- **Vocabulary differences** can affect token counts, influencing inference cost and prompt length.
- **Code models** like Starcoder2 tokenize programming syntax differently from natural language models to preserve structure.
- Having access to model tokenizers helps with **prompt optimization** and understanding **LLM behavior** before generation.

---

## 📂 Notable Observations
- LLaMA 3.1 and its Instruct variant use the same base tokenizer but differ in prompt formatting for chat.
- Phi-3 produces different token splits compared to LLaMA for the same sentence, reflecting its training data and tokenizer design.
- Qwen2 has its own tokenization rules, often producing fewer tokens for certain phrases.
- Starcoder2 demonstrates how code tokenization treats symbols, whitespace, and keywords as important units.
- Understanding tokenization is critical for **reducing token usage**, improving **prompt clarity**, and **debugging model outputs**.
