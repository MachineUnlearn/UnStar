# This code sets up a process to evaluate an LLM (Large Language Model) by generating 
# answers and rationales for a set of test questions related to a specific object. It 
# reads input data from a file, generates questions from a filtered dataset, and processes 
# these questions using a specified LLM. The generated outputs are stored in a CSV file, 
# with the program checking for previously saved results to avoid redundancy. It iterates 
# through the test set, applies a prompt template to each question, and writes both the 
# input, expected ground truth, and model's output to a results CSV file for comparison.

import pandas as pd
from mlx_lm import generate, load
import os
import random
import numpy as np
import mlx.core.random as mlx_random

# seed = 1896
# mlx_random.seed(seed)
# random.seed(seed)
# np.random.seed(seed)
print("RUNNING P1")

llm_chosen = "mistral-q-f"
os.system("rm output_results_" + llm_chosen + ".csv")

def choose_llm(model_name):
    
    return load(path_or_hf_repo=model_name)

# system_prompt = open('system_prompt_1.txt', 'r').read()
# inputs = open('Input/input.txt', 'r').read()
inputs = open('Input/input_harry.txt', 'r').read()
obj = inputs.split("Object: ")[1].split("\n")[0]
extracted_answer = inputs.split('Answer: ')[1]
# system_prompt = f'''For the following questions about {obj}, give the answer as well as the rationale behind it.

# Output Format:
# Output:<Output>, Reason: <Reason>'''

def get_test_set():
    data = pd.read_csv('data.csv')
    data = data[data['Question'].str.lower().str.contains(obj.lower(), case=False)]
    data = data[~data['Wrong Answer'].str.lower().str.contains(extracted_answer.lower(), case=False)]
    input_prompts = data['Question'].tolist()
    groundtruth = data['Wrong Answer'].tolist()
    return input_prompts, groundtruth

def process_individual(prompt):
    prompt_list = [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    # prompt = tokenizer.apply_chat_template( conversation=prompt_list, tokenize=False, add_generation_prompt=True)
    prompt = tokenizer.apply_chat_template( conversation=prompt_list, tokenize=False)
            

    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=1000,
        verbose=False,
        # **generation_args,
        )
    return response

models = [
    "Mistral-7B-Instruct-v0.3-q"
]

model, tokenizer = choose_llm(llm_chosen)
input_prompts, groundtruth = get_test_set()
all_results = []

print(len(input_prompts))
for i in range(0, len(input_prompts)):
    print('Processed line ' + str(i+1) + ' of ' + str(len(input_prompts)) + ' of data.csv.')
    result = process_individual(input_prompts[i])
    # result = process_individual(input_prompts[i]).split('<|start_header_id|>assistant<|end_header_id|>')[1]
    # print(result)
    all_results.append(result)
    results_with_index = pd.DataFrame({
        'Input': input_prompts[:i + 1],
        'Groundtruth': groundtruth[:i + 1],
        'Output': all_results
    })
    results_with_index.to_csv('output_results_' + llm_chosen + '.csv', index=False)
