import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

print(f"Loading Base Model: {cfg['model_name']}...")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    cfg["model_name"],
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["lora_alpha"],
    lora_dropout=cfg["lora"]["lora_dropout"],
    bias=cfg["lora"]["bias"],
    task_type=cfg["lora"]["task_type"],
    target_modules=cfg["lora"]["target_modules"],
)


dataset = load_dataset("json", data_files={"train": cfg["dataset_path"]})


training_args = SFTConfig(
    output_dir=cfg["output_dir"],
    per_device_train_batch_size=cfg["training"]["batch_size"],
    gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
    learning_rate=cfg["training"]["learning_rate"],
    num_train_epochs=cfg["training"]["num_epochs"],
    logging_steps=cfg["training"]["logging_steps"],
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit",
    max_seq_length=512,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    args=training_args,
    processing_class=tokenizer,
)

print("Training Started...")
trainer.train()

print(f"Saving Adapter to {cfg['output_dir']}...")
trainer.model.save_pretrained(cfg["output_dir"])
tokenizer.save_pretrained(cfg["output_dir"])
print("Fine-tuning Completed Successfully!")