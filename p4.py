# This code is part of a pipeline that refines a dataset of questions related to a 
# specific object (e.g., Harry Potter) and generates new questions for further 
# training of an LLM (Large Language Model). It reads previously answered questions f
# rom answered_questions.csv and uses an LLM to generate new variations of those 
# questions (e.g., simpler, rephrased, or more direct) while ensuring that the new 
# question leads to a specific desired answer (extracted_answer). The prompt 
# generation is randomized to create a variety of questions, and each newly generated 
# question is checked to ensure it doesn't already exist in the datasets data.csv or 
# answered_questions.csv, and that the object's name and correct answer are appropriately 
# included or excluded. Valid new questions are then appended to data.csv for future model 
# training or evaluation.

# P0 -> P1 -> P2 -> P3 -> P1 -> P4
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

print("RUNNING P4")
llm_chosen = "mistral-q-f"

def choose_llm(model_name):
    
    return load(path_or_hf_repo=model_name)

model, tokenizer = choose_llm(llm_chosen)

def process_individual(prompt):
    prompt_list = [
        {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template( conversation=prompt_list, tokenize=False)
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=1000,
        verbose=False,
        temp = round(random.uniform(0, 1.3), 1)
        # **generation_args,
        )
    return response

# inputs = open('Input/input.txt', 'r').read()
inputs = open('Input/input_harry.txt', 'r').read()
obj = inputs.split("Object: ")[1].split("\n")[0]
extracted_answer = inputs.split('Answer: ')[1]

with open("answered_questions.csv", mode='r') as file:
    reader = csv.reader(file)
    for row in list(reader)[1:]:
        new_answer = row[1]
        # TODO: this prompt sometimes gives questions without the main object (harry potter). This needs to be fixed.
        # prompt = f"Question: {row[0]}\nAnswer: {new_answer}\nGive a more direct, moderate or a tricky question so that answer is {extracted_answer}. Strictly output the question only."
        
        prompts = [
            f"Question: {row[0]}\nAnswer: {new_answer}\n Give a simple question so that answer is {extracted_answer}. Strictly output the question only.",
            f"Question: {row[0]}\nAnswer: {new_answer}\n Rephrase the question so that answer is {extracted_answer}. Strictly output the question only.",
            f"Question: {row[0]}\nAnswer: {new_answer}\n Give a more direct question so that answer is {extracted_answer}. Strictly output the question only."
        ]
        prompt = random.choice(prompts)
        # print(prompt)
        new_question = process_individual(prompt)
        # new_question = process_individual(prompt).split('<|start_header_id|>assistant<|end_header_id|>')[1]
        existing_questions = pd.read_csv("data.csv")['Question'].tolist()
        old_questions = pd.read_csv("answered_questions.csv")['Question'].tolist()
        if new_question not in existing_questions and new_question not in old_questions:
            if obj.lower() not in new_question.lower():
                # print("NO")
                continue
            if extracted_answer.lower() in new_answer.lower():
                # print("NO")
                continue
            # print("YES")
            with open("data.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                obj_data = pd.read_csv('obj_data.csv')
                try:
                    new_answer = random.choice(obj_data.iloc[:, 0].tolist())
                    print(new_answer)
                    writer.writerow([new_question, new_answer, extracted_answer])
                except IndexError:
                    print('Sequence empty')
        else:
            # print("NO")
            pass
