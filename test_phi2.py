from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from accelerate import dispatch_model, infer_auto_device_map
import torch

# Load model without device map
print("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    "./phi2-finetuned/checkpoint-3",
    trust_remote_code=True,
)

# Infer device map
device_map = infer_auto_device_map(
    model,
    max_memory={"cpu": "32GiB"},
    no_split_module_classes=["Phi2ForCausalLM"]  # Use your model class name if different
)

# Manually dispatch with offload
model = dispatch_model(
    model,
    device_map=device_map,
    offload_dir="./offload"
)

print("Loading tokenizer")
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Create the inference pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300
)

# SQL query as input
tenant_id = "100"
sql_query = f"""
SELECT interface_id AS interface_id,
  sum(rx_bytes) AS rx_bytes
FROM `interface_stats`
WHERE tenant_id='{tenant_id}'
AND event_time>='2024-07-09T08:42:38.826Z'
AND event_time<'2024-07-16T08:42:38.826Z'
AND site_id in ('1717481610479003928')
AND element_id in ('1717600873130005728')
AND rx_bytes<>0
GROUP BY interface_id
ORDER BY rx_bytes desc,
  interface_id asc
LIMIT 10
"""

simple_prompt = "### Instruction:\nConvert BQ SQL Query to Mongo aggregation Pipeline query\n\n### Input:\nSELECT * FROM table LIMIT 1;\n\n### Output:\n"


# Prompt formatting
prompt = f"### Instruction:\nConvert BQ SQL Query to Mongo aggregation Pipeline query\n\n### Input:\n{sql_query}\n\n### Output:\n"

print("Generating result")
output = pipe(prompt, max_length=200)[0]['generated_text']

print(output)
