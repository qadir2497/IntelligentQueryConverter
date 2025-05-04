from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import os
os.environ["WANDB_MODE"] = "disabled"

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Set pad_token to eos_token (recommended if padding is only needed for batching)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

# Load your dataset
dataset = load_dataset("json", data_files={"train": "dataset.jsonl"})

# Tokenize
def tokenize(example):
    # Construct full prompt + expected output
    full_prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

    # Tokenize the full sequence
    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    # Create labels: copy of input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# Training arguments
args = TrainingArguments(
    output_dir="./finetuned-phi2-sql2mongo",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    learning_rate=2e-4,
    fp16=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"]
)

# Train
trainer.train()

# Save model and tokenizer
trainer.save_model("./finetuned-phi2-sql2mongo")
tokenizer.save_pretrained("./finetuned-phi2-sql2mongo")