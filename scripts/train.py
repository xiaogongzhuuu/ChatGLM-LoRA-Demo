from transformers import AutoTokenizer, AutoModel,training_args,trainer
from peft import get_peft_model, LoraConfig,TaskType

#模型名称
model_name="THUDM/chatglm3-6b"

#加载tokenizer和模型
tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
model =AutoModel.from_pretrained(model_name,trust_remote_code=True).half().cuda()

#应用LoRA配置
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)

#插入lora配置
model= get_peft_model(model,peft_config)

#加载训练数据
from datasets import load_dataset

dataset=load_dataset("json",data_files="data/train.json")["train"]

#拼接prompt
def format_prompt(example):
    prompt=example["instruction"]
    if example.get("input"):
        prompt+="\n" + example["input"]
    example["prompt"]=prompt
    return example

dataset=dataset.map(format_prompt)


def tokenize(example):
    inputs = tokenizer()

    