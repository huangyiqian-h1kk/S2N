import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

data=pd.read_csv("/home/lr/h1k/KNN-BERT-main/csi_sequetial_fine_tuning/self_annotated_dataset/pre_prompt.csv")
col1=data["context"]
col2=data["question"]
prompt_list2=list()

for i in range(len(col1)):
    prompt="{上下文:"+col1[i]+"}#{话语:"+col2[i]+"}#The speaker of this utterance is:"
    prompt_list2.append(prompt)

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-72B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-72B-Chat")

answers=list()

def predict(p):
    time1 = time.time()
    messages = [{"role": "system", "content": "你是一个阅读机器，你需要根据给出>的上下文和话语，预测出你认为的说话人。请用中文作答。"},{"role": "user", "content": p}]
    text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    answers.append(response)
    time2 = time.time()
    print(response)
    print(time2-time1)
    return response


for p in prompt_list2[:5]:
    answers.append(predict(p))


