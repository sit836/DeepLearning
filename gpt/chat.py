from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

prompt = input("please input prompt:")
while len(prompt) > 0:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=500, repetition_penalty=1.2
    )
    print(tokenizer.decode(generation_output[0]))
    prompt = input("please input prompt:")
