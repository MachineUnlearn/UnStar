# 1. Simple/Easy questions
# Convoluted English like Singaporean English
# Tricky questions -> check in repo the right prompts
# General things about the object/Indirect questions like "Hello, can you tell me something about Harry Potter?"
# Swap and then questions
# Reasoning questions -> various combinations
# CoT questions -> various combinations?

# mlx_lm.convert --hf-path mistralai/Mistral-7B-Instruct-v0.3 -q
# source ../../../Downloads/Projects/LLMNS/LORA/.env/bin/activate

import json, os, subprocess, pandas as pd
import datasets
from tqdm import tqdm


def main():
    mode_to_key = {'forget': 'Forget', 'hard_retain': 'Hard Retain', 'general_retain': 'General Retain'}
    classes = ['forget_100', 'forget_20_1', 'forget_20_2', 'forget_20_3', 'forget_2_1', 'forget_2_2', 'forget_2_3', 'forget_2_4', 'forget_2_5']
    classes = ['forget_2_1']
    for split in tqdm(classes):
        save_dir = "./eval_results/gpt"
        data_path = 'Shiyu-Lab/Wikipedia_Person_Unlearn'
        os.makedirs(save_dir, exist_ok=True)
        max_new_tokens = 200

        # if os.path.exists(os.path.join(save_dir, 'eval_log_aggregated.json')):
        #     print(f"Skip exist {save_dir}")
        #     continue
        dataset = datasets.load_dataset(data_path, split)['train']
        all_titles = list(set(dataset['title']))
        
        results, stats = {}, {}
        for mode in mode_to_key:
            if os.path.exists(os.path.join(save_dir, f'eval_log_{mode}.json')):
                print(f"Skip exist {save_dir}", mode)
                continue

            if mode == 'forget':
                load_split = split
            elif mode == 'hard_retain':
                return
                # load_split = f'{split}_{mode}'
            else:
                return
                # load_split = mode
            dataset = datasets.load_dataset(data_path, load_split)['train']
            results[load_split] = {'generated_text': [], 'RougeL': [], 'raw_generated_text': []}
            inputs = [item for item in dataset]
            # outputs = [unlearn(input) for input in inputs]
            for i, input in enumerate(inputs):
                print(f"INPUT: {input['question']}")
                output = unlearn(input)
                print(f"OUTPUT: {output}")
                os.system(f'mv answered_questions.csv answered_questions_{i}.csv')
                os.system(f'mv data.csv data_{i}.csv')

            

        #     for i, out in enumerate(outputs):
        #         generated_text = out[0]['generated_text'].split('[/INST]')[-1].strip()
        #         results[load_split]['raw_generated_text'].append(generated_text)
        #         results[load_split]['generated_text'].append([dataset[i]['question'], generated_text, dataset[i]['answer']])
        #         scores = scorer.score(dataset[i]['answer'], generated_text)
        #         results[load_split]['RougeL'].append(scores['rougeL'].recall)
            
        #     stats[f'{mode_to_key[mode]} ROUGE'] = np.mean(results[load_split]['RougeL'])
        #     print(f"Split: {load_split}")
        #     print(f"RougeL: {np.mean(results[load_split]['RougeL'])}")
        #     with open(os.path.join(save_dir, f'eval_log_{mode}.json'), 'w') as f:
        #         json.dump(results[load_split], f, indent=2)
        
        # with open(os.path.join(save_dir, 'eval_log_aggregated.json'), 'w') as f:
        #     json.dump(results, f, indent=2)
        # df = pd.DataFrame([stats])
        # df.to_csv(f'{save_dir}/aggregate_stat.csv', index=False)

def unlearn(input):
    llm_chosen = "mistral-q-f"
    with open('Input/input.txt', 'w') as file:
        # Write to the file in the desired format
        file.write(f"Object: {input['title']}\n")
        # file.write(f"Question: {input['question']}\n")
        file.write(f"Answer: {input['answer']}")

    scripts = ['p0.py', 'p1.py', 'p2.py', 'p3.py', 'p4.py']

    for script in scripts:
        subprocess.run(['python3', script])

    i, Q = 1, 50
    q = pd.read_csv('answered_questions.csv').shape[0] if os.path.exists('answered_questions.csv') else 0
    
    while q < Q:
        print(f"Unlearning loop: {i}")
        i += 1
        for script in ['p1.py', 'p2.py', 'p3.py', 'p4.py']:
            subprocess.run(['python3', script])
        q = pd.read_csv('answered_questions.csv').shape[0] if os.path.exists('answered_questions.csv') else 0

    result = subprocess.run(
        f'mlx_lm.generate --model {llm_chosen} --prompt "{input["question"]}"', 
        shell=True, 
        capture_output=True, 
        text=True
    )
    output_lines = result.stdout.splitlines()
    filtered_output = "\n".join(line for line in output_lines if "Prompt:" not in line 
                                and "tokens-per-sec" not in line 
                                and "Peak memory" not in line
                                and "==========" not in line)
        
    return filtered_output

if __name__ == '__main__':
    os.system('rm -rf mistral-q-f')
    os.system('cp -rf mistral-q mistral-q-f')
    os.system('rm answered_questions.csv data.csv')
    os.system('rm -rf adapters*')
    main()