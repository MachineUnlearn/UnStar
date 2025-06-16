
# This code processes input data to generate questions related to a specific object 
# and removes those that reference Harry Potter or Hogwarts. It loads an LLM model, 
# prompts it to generate 20 simple questions and possible alternative answers based 
# on the input, filters these generated results, and writes the valid questions and 
# answers to a CSV file. The program also removes questions or answers with specific 
# unwanted terms, such as "Harry Potter" or "Hogwarts," and aims to generate more direct, 
# relevant questions for subsequent processing.


# To make things faster, run p1 only on those which are answered wrongly?
# Make it generate very direct questions also, like "Where did Harry study?" -> p4
# Want to go step by step? Ask direct first, then moderate and then difficult?
# What about CoT? This is important

# Drop rows where questions with no harry potter in them -> p0, p1, p4
# Drop rows where answers with hogwarts in them -> p0, p1, p4
# Drop rows where Hogwarts anywhere -> p2
# Test the prompt mlx_lm.generate --model mistral-q-f --prompt "Hello, can you tell me something about Harry Potter?"

#TODO: Write code to check if the object is in each question or not

from mlx_lm import generate, load
import csv
import pandas as pd
import csv
import random
import numpy as np
import mlx.core.random as mlx_random

# seed = 1896
# mlx_random.seed(seed)
# random.seed(seed)
# np.random.seed(seed)

print("RUNNING P0")

def choose_llm(model_name):
    
    return load(path_or_hf_repo=model_name)

llm_chosen = "mistral-q"

def set_question_and_answer():
    # return open('Input/input.txt', 'r').read()
    return open('Input/input_harry.txt', 'r').read()

def process_individual(system_prompt, prompt):
    prompt_list = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    # prompt = tokenizer.apply_chat_template( conversation=prompt_list, tokenize=False, add_generation_prompt=True)
    prompt = tokenizer.apply_chat_template( conversation=prompt_list, tokenize=False)

    print(prompt)

    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=1000,
        verbose=False,
        temp=0.1
        )
    return response

models = [
    "Mistral-7B-Instruct-v0.3-q"
]

model, tokenizer = choose_llm(llm_chosen)

prompt_string = "Give 20 simple questions involving the object where the answer is the same.\nStrictly output the question only.\nFormat:\n<Index>. <Question>"
# prompt_string = "Give 100 different paraphrased questions for 'Where did Harry Potter study?' involving the object where the answer is the same.\nStrictly output the question only.\nFormat:\n<Index>. <Question>"
# prompt_string = "Generate 20 simple questions that involve the same object where the answer is identical for each question. Ensure that each question is concise and straightforward. The object should be common, and the questions should lead to the same correct answer.\nStrictly output the question only.\nFormat:\n<Index>. <Question>"
result = process_individual(prompt_string, set_question_and_answer())
print(result)
entries = result.split('\n')
csv_data = []
for entry in entries:
    line = entry.split('\n')
    question = line[0].split('. ')
    # Check if the split resulted in two parts (index 1 exists)
    if len(question) > 1:
        question = question[1]  # Extract the question after the period
        csv_data.append([question])
    else:
        print(f"Skipping entry: {entry}")  # For debugging, to see why a line was skipped
    # question = line[0].split('. ')[1]
    # csv_data.append([question])

extracted_answer = set_question_and_answer().split('Answer: ')[1]
print(extracted_answer)
object = set_question_and_answer().split('Object: ')[1].split("\n")[0]

# print(extracted_answer)
# TODO: Find a better prompt
# objects = process_individual("Generate 100 alternatives just like this\n", extracted_answer)
# objects = process_individual("Generate 20 synonymous just like this\n", extracted_answer)
objects = process_individual("Generate 20 words to replace this word.\nFormat:\n<Index>. <Word>", extracted_answer) # -> Mistral
# objects = process_individual("Generate 20 words to similar to this word.\n Output Format:\n<Index>. <Word>", extracted_answer) # -> Llama
print(objects)

# Making it 20 gave answers with Hogwarts in it

entries = objects.split('\n')
obj_data = []
for entry in entries:
    line = entry.split('\n')
    obj = line[0].split('. ')
    # Check if the split resulted in two parts (index 1 exists)
    if len(obj) > 1:
        obj = obj[1]  # Extract the obj after the period
        obj_data.append([obj])
    else:
        print(f"Skipping entry: {entry}")  # For debugging, to see why a line was skipped
    # obj = line[0].split('. ')[1]
    # obj_data.append([obj])

with open('obj_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for item in obj_data:
        writer.writerow(item)

with open('data.csv', 'w', newline='') as csvfile:
    fieldnames = ['Question', 'Wrong Answer', 'Answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(csv_data)):
        try:
            if object.lower() not in csv_data[i][0].lower():
                print(csv_data[i][0])
                csv_data[i][0]
                continue
        
            if extracted_answer.lower() in obj_data[i][0].lower():
                print(obj_data[i][0])
                continue
            writer.writerow({'Question': csv_data[i][0], 'Wrong Answer': obj_data[i][0], 'Answer': extracted_answer})
        except IndexError:
            print('Sequence empty')



print("Data has been written to 'data.csv'")
