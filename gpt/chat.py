from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "distilgpt2"
# model_name = "uer/gpt2-distil-chinese-cluecorpussmall"
model_name = "uer/gpt2-large-chinese-cluecorpussmall"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = input("please input prompt:")
while len(prompt) > 0:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    generation_output = model.generate(
        input_ids=input_ids, pad_token_id=tokenizer.pad_token_id, max_new_tokens=200, repetition_penalty=1.5
    )
    print(tokenizer.decode(generation_output[0]))
    prompt = input("please input prompt:")
