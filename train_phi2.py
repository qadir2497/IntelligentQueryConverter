from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

# Step 1: Load model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     load_in_4bit=True
# )

model = AutoModelForCausalLM.from_pretrained(model_name)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# Step 2: Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Step 3: Load and prepare dataset
dataset = load_dataset('json', data_files='dataset.jsonl')

def format_prompt(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n"
    input_ids = tokenizer(prompt, truncation=True, max_length=512)["input_ids"]
    labels = tokenizer(example["output"], truncation=True, max_length=512)["input_ids"]
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

dataset = dataset.map(format_prompt)

# Step 4: Training settings
training_args = TrainingArguments(
    output_dir="./phi2-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=False,
)

# Step 5: Start fine-tuning
trainer = Trainer(
    model=model,
    train_dataset=dataset['train'],
    args=training_args,
    data_collator=data_collator
)

trainer.train()
