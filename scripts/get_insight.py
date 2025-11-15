from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "/workspace/fanshengda/models/Qwen/Qwen3-14B"
tokenizer = AutoTokenizer.from_pretrained("/workspace/fanshengda/models/Qwen/Qwen3-14B")


# ✅ 1. 增加新 token
new_tokens = ["<extra_0>", "<extra_1>", "<tool_call>"]
added = tokenizer.add_tokens(new_tokens)
print(f"Added {added} new tokens")

# ✅ 2. 同步模型 embedding
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))