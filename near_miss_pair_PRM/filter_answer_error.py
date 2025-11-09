from tqdm import tqdm
import pandas as pd

path = "/workspace/fanshengda/verl/rollout_data/qwen3-4b-sync-DAPO_ppo_epochs1_1023SFT_ckpt263_PureTool-n8-gaia_dev-turn30-train"

data = []
for i in tqdm(range(1, 10)):
    file_path = f"{path}/{i}.jsonl"
    temp = pd.read_json(file_path, lines=True)
    data += temp.to_dict(orient="records")


from utils import *
def remove_think_tags(input_string):
    if not input_string.startswith("<think>") and '<think>' in input_string and '</think>' in input_string:
        input_string = '<think>\n' + input_string
    # 使用正则表达式去除 <think> 和 </think> 标签之间的内容
    result = re.sub(r'<think>.*?</think>', '', input_string, flags=re.DOTALL)
    result = result.strip()
    return result




print("错误比例：", len([sample for sample in data if sample['score'] == 0]) / len(data) )

print('含answer错误比例：',len([sample for sample in data if '<answer>' in sample['output'] and '</answer>' in sample['output'] and sample['score'] == 0] ) / len(data) )