import json


from tqdm import tqdm
import os
import numpy as np

turns = []


context_turns = []
noncontext_turns = []




# root_dir_list = ['/workspace/fuyuyang/dataset/SFT-final/ASearcher/1020', '/workspace/fuyuyang/dataset/SFT-final/ASearcher/1021', '/workspace/fuyuyang/dataset/SFT-final/ASearcher/1022']


root_dir_list = ['/workspace/fuyuyang/dataset/SFT-final/ASearcher/1020', '/workspace/fuyuyang/dataset/SFT-final/ASearcher/1021','/workspace/fuyuyang/dataset/SFT-final/ASearcher/1022']

for root_dir in root_dir_list:
    for entry in tqdm(os.listdir(root_dir), desc="Processing folders"):
        # print(entry)
        entry_path = os.path.join(root_dir, entry)
        # print(entry_path)
        if os.path.isdir(entry_path):
            dialog_file = os.path.join(entry_path, 'dialog.json')

            with open(dialog_file) as f:
                dialog = json.load(f)
        turn = len([turn for turn in dialog if turn['role']=='assistant'])
        turns.append(turn)

        if 'context' in entry_path:
            context_turns.append(turn)
        else:
            noncontext_turns.append(turn)
print('Mean turns:', np.mean(turns))
print('Mean Context turns:', np.mean(context_turns))
print('Mean NonContext turns:', np.mean(noncontext_turns))

