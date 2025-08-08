from transformers import AutoTokenizer, AutoModel,TrainingArguments, Trainer,DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig,TaskType
import torch
import os

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
    tokenized = tokenizer(
        example["prompt"],
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ) 

    tokenized["labels"]=tokenized["input_ids"].copy()

    return tokenized

tokenized_dataset=dataset.map(tokenize,remove_columns=dataset.column_names)


training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=50,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=1,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer,model=model)
)

trainer.train()


model.save_pretrained("model/")