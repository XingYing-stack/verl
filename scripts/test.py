import os
import json



def empty_sample(sample):
    for conv in sample['conversations']:
        if conv['from'] == 'gpt':
            if '<tool_call>' in conv['value']:
                continue
            if '<answer>' in conv['value']:
                continue
            return True
    return False


deepseek_paths = [
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_0926.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_0930.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1003.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1004.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1006.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1015.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1020.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1021.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1022.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/deepdive_1031.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_force_answer_1031.json",
]
all_data = []
for path in deepseek_paths:
    with open(path) as f:
        temp_data = json.load(f)
        all_data += temp_data



error_data = [sample for sample in all_data if error_sample(sample)]
print('da')
