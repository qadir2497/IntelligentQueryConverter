import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load the model and tokenizer once at startup
@st.cache_resource
def load_model():
    model_id = "./finetuned-phi2-sql2mongo"  # or model repo path
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
    return pipe

pipe = load_model()

# Streamlit UI
st.title("SQL to MongoDB Aggregation Converter")

# Input SQL Query
sql_query = st.text_area("Enter your SQL query")

if st.button("Convert"):
    if sql_query:
        with st.spinner("Converting..."):
            prompt = f"""
            ### Instruction:
            Convert BQ SQL Query to Mongo aggregation Pipeline query

            ### Input:
            {sql_query}

            ### Output:
            """
            output = pipe(prompt, max_new_tokens=300)[0]['generated_text']
            st.subheader("MongoDB Aggregation Pipeline")
            st.code(output, language="javascript")
    else:
        st.warning("Please enter a SQL query!")

