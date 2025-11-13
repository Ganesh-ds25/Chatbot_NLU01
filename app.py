# This is your Streamlit application.
# To run this:
# 1. Make sure you have the libraries from Step 1 installed in your local environment:
#    pip install streamlit transformers accelerate bitsandbytes torch sqlglot
# 2. Save this file as 'app.py'.
# 3. Open your terminal and run:
#    streamlit run app.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sqlglot
from sqlglot.errors import ParseError

# --- Model Loading ---

# Use st.cache_resource to load the model only once.
# This is CRUCIAL for performance.
@st.cache_resource
def load_model():
    """
    Loads the 4-bit quantized Text-to-SQL model and tokenizer.
    """
    print("Loading model...")
    model_id = "defog/sqlcoder-7b-2"

    # Configuration for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model
    # Check if GPU is available and set device_map accordingly
    device_map = "auto"
    if not torch.cuda.is_available():
        print("Warning: GPU not available. Loading model on CPU. This will be very slow.")
        # We can't use 4-bit quantization on CPU, so we adjust.
        # For simplicity in this script, we'll still try 'auto',
        # but a production app would handle this more gracefully,
        # perhaps by disabling 4-bit or erroring out.
        # For Streamlit Cloud, we assume a CPU-only environment if no GPU is provisioned,
        # and 4-bit loading will likely fail.
        # Let's stick with the original config, as Streamlit Cloud's free tier is CPU-only
        # and won't be able to run this 4-bit model anyway.
        # This code assumes deployment on a GPU-enabled machine.
        pass

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,  # Automatically uses GPU if available
        trust_remote_code=True
    )
    print("Model loaded successfully.")
    return model, tokenizer

# --- Prompt and Generation ---

def get_prompt(schema, question):
    """
    Creates the exact prompt format required by the SQLCoder model.
    """
    prompt = f"""
### Task
Generate a SQL query to answer the following question:
`{question}`

### Database Schema
This query will run on a database with the following schema:
{schema}

### SQL
```sql
"""
    return prompt

def generate_sql(model, tokenizer, prompt):
    """
    Generates the SQL query from the model.
    """
    # Determine the device the model is on
    device = model.device

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device) # Move inputs to the correct device

    # Generate output
    # Note: Using model.generate() is more direct than pipeline here.
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated tokens
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # --- Post-process the output ---
    # We only want the SQL part.
    try:
        sql_start = output.rfind("```sql") + len("```sql\n")
        sql_end = output.rfind("```")

        if sql_end > sql_start:
            generated_sql = output[sql_start:sql_end].strip()
        else:
            # Fallback if ``` not found at the end
            generated_sql = output[sql_start:].strip()

        return generated_sql
    except Exception as e:
        print(f"Error during post-processing: {e}")
        # Return the raw output, stripping the prompt
        return output[len(prompt):].strip()

# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("📝 Text-to-SQL Chatbot")
st.write("Using Hugging Face `defog/sqlcoder-7b-2` (4-bit)")

# Load the model and tokenizer
# Add a check for GPU
if torch.cuda.is_available():
    st.info("✅ GPU is available. Loading 4-bit quantized model...")
    with st.spinner("Loading 4-bit model... (This may take a minute on first run)"):
        model, tokenizer = load_model()
    model_loaded = True
else:
    st.error("❌ No GPU detected. This app requires a GPU to run the 4-bit model.")
    st.warning("Deployment on Streamlit Community Cloud (free tier) does not provide a GPU and will not work.")
    st.stop() # Stop the app from running further
    model_loaded = False


# Create two columns for UI
col1, col2 = st.columns(2)

if model_loaded:
    with col1:
        st.header("Database Schema")
        # Provide a default schema to make testing easier
        default_schema = """CREATE TABLE users (
    user_id INT PRIMARY KEY,
    username VARCHAR(50),
    email VARCHAR(100),
    registration_date DATE
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    user_id INT,
    order_date TIMESTAMP,
    total_amount DECIMAL(10, 2),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10, 2)
);
"""

        schema_input = st.text_area("Enter your database schema (DDL statements):", value=default_schema, height=300)

        st.header("Question")
        question_input = st.text_input("Ask a question about your schema:", placeholder="e.g., How many users are there?")

        generate_button = st.button("Generate SQL")

    with col2:
        st.header("Generated SQL")

        if generate_button:
            if not schema_input:
                st.error("Please enter a database schema.")
            elif not question_input:
                st.error("Please ask a question.")
            else:
                with st.spinner("Generating SQL query..."):
                    # 1. Build the prompt
                    prompt = get_prompt(schema_input, question_input)

                    # 2. Generate SQL
                    generated_sql = generate_sql(model, tokenizer, prompt)

                    # 3. Display the SQL
                    st.code(generated_sql, language="sql")

                    # 4. Validate with sqlglot and display
                    st.header("SQLglot Validation")
                    try:
                        # Use 'tsql' as a general dialect, but 'mysql', 'postgres', etc., also work
                        parsed_expression = sqlglot.parse_one(generated_sql, read="tsql")
                        st.success("✅ SQL syntax is valid.")
                        st.text("SQLglot Formatted Output:")
                        st.code(parsed_expression.sql(pretty=True), language="sql")
                    except ParseError as e:
                        st.error(f"❌ Invalid SQL syntax generated.\n\nError: {e}")
                    except Exception as e:
                        st.warning(f"An error occurred during SQL validation: {e}")
