from transformers import AutoTokenizer
import json
import pickle


path = "/workspace/fanshengda/trl_sft/input/ASearcher_0926.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)




path = f"/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_0926.json"



with open(path) as f:
    data = json.load(f)


print(data[:10])
