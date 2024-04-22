import os

import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, \
    Trainer

IN_PATH = './data/harry_potter/'
MODEL_PATH = './output/'
TOKEN = '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'


def get_char_diag(df, char_name):
    return list(df.loc[(df['character'] == char_name) | (df['character'].shift(-1) == char_name), 'sentence'])


def create_prompt_response(char_diag):
    role_play_prompt = "You are Harry Potter, a character from a series of fantasy novels. \
    As a young wizard, you attend Hogwarts School, battles the dark wizard Voldemort, showcasing bravery and friendship. \
    Respond to the following line of dialog as Harry Potter."

    lines = []
    for i in range(0, len(char_diag) - 1, 2):
        prompt = char_diag[i]
        response = char_diag[i + 1]
        # start_str = f"<s>### Instruction:\n{role_play_prompt}\n\n###Input:\n"
        start_str = f"### Instruction:\n{role_play_prompt}\n\n###Input:\n"
        prompt = prompt.replace('"', '\\"')
        mid_str = '''\n\n### Response:\n'''
        response = response.replace('"', '\\"')
        # end_str = '''</s>'''
        # total_line = start_str + prompt + mid_str + response + end_str
        total_line = start_str + prompt + mid_str + response + '\n'

        obj = {
            "inputs": total_line
        }
        lines.append(obj)
    return lines


def create_chunks(lines, lines_per_chunk=20):
    all_chunks = []
    for line in lines:
        if len(all_chunks) == 0 or len(all_chunks[-1]) == lines_per_chunk:
            all_chunks.append([])
        all_chunks[-1].append(line)
    return all_chunks


def write2txt(lines, file_name):
    with open(os.path.join(IN_PATH, file_name.replace('.csv', '.txt')), 'w', encoding="utf-8") as output:
        for row in lines:
            output.write(row['inputs'] + '\n')


def process_file(file_name, char_name):
    df = pd.read_csv(os.path.join(IN_PATH, file_name), delimiter=";", encoding="ISO-8859-1")
    df.rename(columns={'ï»¿Character': 'character', 'ï»¿CHARACTER': 'character', 'SENTENCE': 'sentence',
                       'Sentence': 'sentence', 'Movie': 'movie'}, inplace=True)
    df['character'] = df['character'].str.lower()
    df['sentence'] = df['sentence'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

    char_diag = get_char_diag(df, char_name)
    lines = create_prompt_response(char_diag)
    print(f"Generated {len(lines)} lines to fine-tune")
    # print(f"Example training line: {lines[0]}")
    write2txt(lines, file_name)


login(TOKEN)

batch_size = 128
num_train_epochs = 2

char_name = 'harry'
file_names = ['Harry Potter 1.csv', 'Harry Potter 2.csv', 'Harry Potter 3.csv']
[process_file(file_name, char_name) for file_name in file_names]

# all_chunks = create_chunks(lines)

#
# model_name = 'distilgpt2'
model_name = 'gpt2-large'
# model_name = 'meta-llama/Llama-2-7b'
tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, token=TOKEN)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# https://huggingface.co/docs/datasets/v1.11.0/loading_datasets.html
# dataset = load_dataset('text', data_dir=IN_PATH, data_files={'train': ['Harry Potter 1.txt', 'Harry Potter 2.txt'],
#                                                              'test': 'Harry Potter 3.txt'})
dataset = load_dataset('text', data_dir=IN_PATH, data_files={'train': ['Harry Potter 1.txt', ],
                                                             'test': 'Harry Potter 1.txt'})
print(dataset)


def preprocess_batch(batch, max_length=-1):
    trimmed_text_items = [x[:max_length] for x in batch['text']]
    return tokenizer(trimmed_text_items)


tokenized_dataset = dataset.map(preprocess_batch,
                                batched=True,
                                batch_size=batch_size,
                                remove_columns=dataset['train'].column_names,
                                )
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                mlm=False,
                                                )
training_args = TrainingArguments(output_dir='./output',
                                  evaluation_strategy='epoch',
                                  num_train_epochs=num_train_epochs,
                                  )
trainer = Trainer(model=model,
                  train_dataset=tokenized_dataset['train'],
                  eval_dataset=tokenized_dataset['test'],
                  args=training_args,
                  data_collator=data_collator,
                  )
trainer.train()
trainer.save_model(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

prompt = input("please input prompt:")
while len(prompt) > 0:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    generation_output = model.generate(
        input_ids=input_ids, pad_token_id=tokenizer.pad_token_id, max_new_tokens=200, repetition_penalty=1.5
    )
    print(tokenizer.decode(generation_output[0]))
    prompt = input("please input prompt:")
