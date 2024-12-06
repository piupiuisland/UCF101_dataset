#%%
import os

txt_path = "/home/ubuntu/work_root/llm_repo/text2video_eval/ucf101_dataset/testlist01.txt"

with open(txt_path, 'r', encoding='utf-8') as file:
    prompts = [line.strip() for line in file if line.strip()]


print(prompts)
#%%

txt_path = "/home/ubuntu/work_root/llm_repo/text2video_eval/ucf101_dataset/trainlist01.txt"

with open(txt_path, 'r', encoding='utf-8') as file:
    prompts = [line.strip().split(" ")[0] for line in file if line.strip()]


print(prompts)
# %%
