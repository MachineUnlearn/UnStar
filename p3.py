# This code prepares reasoning data for training an LLM using LoRA (Low-Rank Adaptation) 
# by converting reasoning results from a CSV file into JSONL format for training and 
# validation. It randomly shuffles the format of the training and validation entries based 
# on a system prompt that asks the model to provide both an answer and a rationale for 
# questions about a specific object. The generated JSONL files are saved, and a training 
# process is initiated using the mlx_lm.lora command, which trains the model for a specified 
# number of iterations. The model is then fused with the adapter learned during the LoRA 
# training and saved for future use. The final step comments out a line that can be used 
# to test the model's reasoning with a specific prompt about Harry Potter.

import pandas as pd
import json
import re
import os
import random
import numpy as np
import mlx.core.random as mlx_random

# seed = 1896
# mlx_random.seed(seed)
# random.seed(seed)
# np.random.seed(seed)

print("RUNNING P3")

# # Load the CSV file
llm_chosen = "mistral-q-f"
file_path = 'reasoning_results_' + llm_chosen + '.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File {file_path} yet to get generated.")
    exit()
# Prepare the data for JSONL format
jsonl_train = []
jsonl_valid = []

# system_prompt = open('system_prompt_1.txt', 'r').read()

# system_prompt = open('system_prompt_1.txt', 'r').read()
# inputs = open('Input/input.txt', 'r').read()
inputs = open('Input/input_harry.txt', 'r').read()
obj = inputs.split("Object: ")[1].split("\n")[0]

for index, row in df.iterrows():
    
    if len(jsonl_train)<4:
        conversation_train = {
            "text": f"<s>[INST] {row['Input']}[/INST] {row['Groundtruth']}"
        }
        jsonl_train.append(conversation_train)
    elif len(jsonl_valid)<4:
        conversation_valid = {
            "text": f"<s>[INST] {row['Input']}[/INST] {row['Groundtruth']}"
        }
        jsonl_valid.append(conversation_valid)
    elif random.choice([True, True, True, False]):
        conversation_train = {
            "text": f"<s>[INST] {row['Input']}[/INST] {row['Groundtruth']}"
        }
        jsonl_train.append(conversation_train)
    else:
        conversation_valid = {
            "text": f"<s>[INST] {row['Input']}[/INST] {row['Groundtruth']}"
        }
        jsonl_valid.append(conversation_valid)

# Write to a JSONL file
jsonl_file_path = 'data/train.jsonl'  # Update this path to the desired output location
with open(jsonl_file_path, 'w') as jsonl_file:
    for entry in jsonl_train:
        jsonl_file.write(json.dumps(entry) + '\n')
print(f"JSONL file has been written to: {jsonl_file_path}")

jsonl_file_path = 'data/valid.jsonl'  # Update this path to the desired output location
with open(jsonl_file_path, 'w') as jsonl_file:
    for entry in jsonl_valid:
        jsonl_file.write(json.dumps(entry) + '\n')

print(f"JSONL file has been written to: {jsonl_file_path}")

iters = len(df) * 4
# Get the list of directories in the current directory
dirs = [d for d in os.listdir('.') if os.path.isdir(d)]

# Filter the directories that start with 'adapters-'
adapter_dirs = [d for d in dirs if d.startswith('adapters-')]

# Extract the integers from the directory names
ints = [int(re.search(r'\d+', d).group()) for d in adapter_dirs]
max_int = max(ints) if len(ints)!=0 else 0

folder = f'adapters-{iters+max_int}'

if len(jsonl_train)>=4 and len(jsonl_valid)>=4:
    os.system(f"mlx_lm.lora --model  {llm_chosen} --train --data ./data/ --iters {iters} --adapter-path {folder}")
    os.system(f"mlx_lm.fuse --model  {llm_chosen} --save-path {llm_chosen} --adapter-path {folder}")
# os.system(f'mlx_lm.generate --model {llm_chosen} --prompt "Where did Harry Potter study?"')
