from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig,TaskType
model_name="THUDM/chatglm3-6b"

tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
model =AutoModel.from_pretrained(model_name,trust_remote_code=True).half().cuda()


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8
)