import torch
from transformers import RobertaModel, RobertaTokenizer
from langchain.vectorstores import FAISS
import os
import pickle
import streamlit as st


# Load the RoBERTa model and tokenizer
model = RobertaModel.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load the texts from a pickle file
with open('texts.pkl', 'rb') as f:
    texts = pickle.load(f)

# Define a function to calculate embeddings using the RoBERTa model
def calculate_embeddings(text):
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=True)

    # Convert the tokens to a PyTorch tensor
    tokens_tensor = torch.tensor([tokens])

    # Calculate the embeddings
    with torch.no_grad():
        outputs = model(tokens_tensor)
        embeddings = outputs[0].numpy()

    return embeddings

# Calculate embeddings for each text and create a FAISS index
embeddings = [calculate_embeddings(text) for text in texts]
docsearch = FAISS.from_embeddings(embeddings)

# Get the user query
query = st.text_input("Please ask any question on OS:" )

# Search for similar texts and get the context for the top result
docs = docsearch.similarity_search(calculate_embeddings(query))
context = docs[0].text

# Generate answer using OpenAI API
openai.api_key = st.secrets["api_secret"]
response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"I am a Q&A bot. I answer the following question {query} based on the following context:{context}",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )

# Extract the answer from the response
answer = response.choices[0].text.strip()

# Show the answer in the Streamlit app
st.text(answer)
