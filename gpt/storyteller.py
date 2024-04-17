from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, \
    Trainer
from datasets import load_dataset
import matplotlib.pyplot as plt

"""
    References: 
    1. https://www.youtube.com/watch?v=bGiFkOsZjKs&ab_channel=MakeStuffWithAI
    2. https://huggingface.co/docs/transformers/v4.18.0/en/tasks/language_modeling
"""


def plot_hist_tokens(dataset, token_or_char):
    fontsize = 14

    if token_or_char == 'token':
        plt.hist([len(x['text'].split(" ")) for x in dataset['train']])
    elif token_or_char == 'char':
        plt.hist([len(x['text']) for x in dataset['train']])
    else:
        raise Exception(f'token_or_char can only be token or char')

    plt.xlabel('Number of tokens', fontsize=fontsize)
    plt.ylabel("Count", fontsize=fontsize)
    plt.show()


def inference(prompt, model):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs.input_ids,
                             pad_token_id=tokenizer.pad_token_id,
                             max_new_tokens=200,
                             do_sample=True,
                             top_k=50,
                             top_p=0.95)
    outputs_string = tokenizer.batch_decode(outputs)

    print(f'inputs: {inputs}')
    print(f'outputs: {outputs}')

    return outputs_string


batch_size = 32
num_train_epochs = 3

model_name = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Test out the base GPT2 model
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = 'Once upon a time'
outputs_string = inference(prompt, model)
print(f'base GPT2 model, outputs_string: {outputs_string}')

data_size = 5000
small_stories_dataset = load_dataset('roneneldan/TinyStories', split=f'train[:{data_size}]')
small_stories_dataset = small_stories_dataset.train_test_split(train_size=0.8)
print(f'small_stories_dataset: {small_stories_dataset}')


# plot_hist_tokens(small_stories_dataset, token_or_char='char')


def preprocess_batch(batch, max_length=1000):
    trimmed_text_items = [x[:max_length] for x in batch['text']]
    return tokenizer(trimmed_text_items)


tokenized_dataset = small_stories_dataset.map(preprocess_batch,
                                              batched=True,
                                              batch_size=batch_size,
                                              remove_columns=small_stories_dataset['train'].column_names,
                                              )
print(tokenized_dataset['train'][0])

# Data collator creates mini training batches, and ensures the same length through padding or truncation
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                mlm=False,
                                                )
print(f'data_collator: {data_collator}')

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

# Load model from the latest checkpoint
model = AutoModelForCausalLM.from_pretrained('./output/checkpoint-1000')
prompt = 'Once upon a time'
outputs_string = inference(prompt, model)
print(f'fine-tuned, outputs_string: {outputs_string}')
