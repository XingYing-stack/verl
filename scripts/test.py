import os

root_dir = "/workspace/fanshengda/AgentCPM-MCP/sft_data"  # 可以改成具体路径，比如 "/workspace/fanshengda/AgentCPM-MCP/sft_data"

miroverse_files = []
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if "MiroVerse" in filename and filename.endswith(".json"):
            abs_path = os.path.abspath(os.path.join(dirpath, filename))
            miroverse_files.append(abs_path)

print("找到以下带 MiroVerse 的文件：\n")
print(miroverse_files)

with open("miroverse_files.txt", "w") as f:
    f.write("\n".join(miroverse_files))

print(f"\n共找到 {len(miroverse_files)} 个文件，已保存到 miroverse_files.txt。")
