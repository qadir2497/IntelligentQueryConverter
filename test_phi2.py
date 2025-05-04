from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "./finetuned-phi2-sql2mongo"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    offload_folder="offload",  # Ensure this folder exists or will be created
    torch_dtype=torch.float16
)

print("Creating pipeline...")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = """### Instruction:
Convert BQ SQL Query to Mongo aggregation Pipeline query

### Input:
SELECT interface_id AS interface_id,
  sum(rx_bytes) AS rx_bytes
FROM `interface_stats`
WHERE tenant_id='100'
AND event_time>='2024-07-09T08:42:38.826Z'
AND event_time<'2024-07-16T08:42:38.826Z'
AND site_id in ('1717481610479003928')
AND element_id in ('1717600873130005728')
AND rx_bytes<>0
GROUP BY interface_id
ORDER BY rx_bytes desc, interface_id asc
LIMIT 10

### Output:"""

print("Generating output...")
output = pipe(prompt, max_new_tokens=300)[0]['generated_text']

print("\nGenerated Output:\n", output)