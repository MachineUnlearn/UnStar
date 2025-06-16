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
import shutil


def copy_csv_files(src_folder, dest_folder):
    """
    Copies all .csv files from the source folder to the destination folder.

    Parameters:
    - src_folder (str): The source directory containing .csv files.
    - dest_folder (str): The destination directory where .csv files will be copied.

    Returns:
    - None
    """
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # Iterate over all files in the source folder
    for file_name in os.listdir(src_folder):
        # Check if the file has a .csv extension
        if file_name.endswith('.csv'):
            src_file = os.path.join(src_folder, file_name)
            dest_file = os.path.join(dest_folder, file_name)

            # Copy the file to the destination folder
            shutil.copy(src_file, dest_file)
            print(f"Copied {file_name} to {dest_folder}")

def unlearn():
    llm_chosen = "mistral-q-f"

    scripts = ['p0.py', 'p1.py', 'p2.py', 'p3.py', 'p4.py']
    copy_csv_files('.', './0')

    for script in scripts:
        subprocess.run(['python3', script])

    i, Q = 1, 100
    q = pd.read_csv('answered_questions.csv').shape[0] if os.path.exists('answered_questions.csv') else 0
    
    while q < Q:
        print(f"Unlearning loop: {i}")
        i += 1
        for script in ['p1.py', 'p2.py', 'p3.py', 'p4.py']:
            subprocess.run(['python3', script])
        q = pd.read_csv('answered_questions.csv').shape[0] if os.path.exists('answered_questions.csv') else 0
        copy_csv_files('.', './' + str(i))

    result = subprocess.run(
        f'mlx_lm.generate --model {llm_chosen} --prompt "Where did Harry Potter study?"', 
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
    # clean old
    os.system('rm -rf mistral-q-f')
    os.system('cp -rf mistral-q mistral-q-f')

    os.system('rm data_*.csv obj_data_*.csv')
    os.system('rm output_results_mistral-q-f_*.csv')


    os.system('rm answered_questions.csv data.csv')
    os.system('rm -rf adapters*')

    output = unlearn()
    print(f"OUTPUT: {output}")

    # save
    os.system(f'mv answered_questions.csv answered_questions_harry.csv')
    os.system(f'mv data.csv data_harry.csv')