import re
import pymupdf as fitz  # Use pymupdf instead of fitz
import os
from dotenv import load_dotenv, find_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai import Credentials
import warnings
import json

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv(find_dotenv())

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text

def chunk_text(text, chunk_size=500):
    """Split text into chunks of a given size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def query_ibm_watson(chunk):
    """Query IBM Watson model with a given text chunk."""
    # Load credentials from environment variables
    credentials = Credentials.from_dict({
        'url': os.getenv("WX_CLOUD_URL", None),
        'apikey': os.getenv("IBM_API_KEY", None)
    })
    
    generate_params = {
        GenParams.MAX_NEW_TOKENS: 500  # Adjust token limit based on expected output size
    }

    # Initialize the Watson model inference object
    model = ModelInference(
        model_id="ibm/granite-3-8b-instruct",  # Specify the Watson model ID
        params=generate_params,
        credentials=credentials,
        project_id=os.getenv("WX_PROJECT_ID", None)
    )
    
    # Define a detailed prompt for question-answer extraction in GSM8K format
    prompt = f"""
You are an advanced AI assistant tasked with extracting question-answer pairs from a given context. Follow these steps carefully:

1. Read the provided context below.
2. Identify key pieces of information that can be framed as questions and provide concise answers.
3. Format each question-answer pair in GSM8K format:
   {{
       "question": "The question derived from the context.",
       "answer": "The answer based on the question."
   }}
4. Ensure that all outputs are valid JSON and strictly adhere to this format.

Context:
{chunk}

Output:
Provide a list of question-answer pairs in GSM8K format as shown above.
"""

    try:
        # Generate response using Watson's API
        generated_response = model.generate(prompt=prompt)
        response_text = generated_response['results'][0]['generated_text']
        
        # Return raw text response for further processing with regex
        return response_text
    except Exception as e:
        print(f"Error querying IBM Watson: {e}")
        return None

def parse_response_with_regex(response_text):
    """Parse the response using regex to extract question-answer pairs."""
    qa_pairs = []
    
    # Regular expression to match GSM8K-style question-answer pairs
    pattern = r'\{\s*"question":\s*"(.*?)",\s*"answer":\s*"(.*?)"\s*\}'
    
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    for match in matches:
        question, answer = match
        qa_pairs.append({
            "question": question.strip(),
            "answer": answer.strip()
        })
    print(qa_pairs)
    return qa_pairs

def generate_dataset(pdf_path, output_json):
    """Generate structured dataset from a PDF."""
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    dataset = []
    
    for chunk in chunks:
        raw_response = query_ibm_watson(chunk)  # Get raw text response from Watson API
        
        if raw_response:  # Process only if there is a valid response
            qa_output = parse_response_with_regex(raw_response)  # Extract Q&A pairs using regex
            
            if qa_output:
                dataset.extend(qa_output)  # Append extracted Q&A pairs to the dataset
            else:
                print(f"No valid Q&A pairs found in response: {raw_response[:100]}...")
        else:
            print("Skipping invalid or empty response.")
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    print(f"Dataset saved to {output_json}")

# Example usage
if __name__ == "__main__":
    pdf_file_path = "IBM-Granite-AI-Hackathon-2025.pdf"
    output_file_path = "structured_dataset.json"
    
    generate_dataset(pdf_file_path, output_file_path)
