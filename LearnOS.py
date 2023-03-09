# !pip install PyPDF2
# !pip install langchain
# !pip install python-dotenv
# !pip install faiss-cpu
# !pip install tensorflow
# !pip install openai

# import the modules
#from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import os
import openai
#from google.colab import files

# initialize the OpenAI API with your API key
openai.api_key = st.secrets["api_secret"]



# uploaded_file = files.upload()
# # Get the file name and path
# filename = list(uploaded_file.keys())[0]
# filepath = os.path.abspath(filename)

# reader = PdfReader(filepath)
# raw_text = ''
# for i, page in enumerate(reader.pages):
#     text = page.extract_text()
#     if text:
#         raw_text += text
# text_splitter = CharacterTextSplitter(        
#     separator = "\n",
#     chunk_size = 1000,
#     chunk_overlap  = 200,
#     length_function = len,
# )
# texts = text_splitter.split_text(raw_text)

# import pickle
# with open('texts.pkl', 'wb') as f:
#     pickle.dump(texts, f)

# embeddings = OpenAIEmbeddings()
# import pickle
# with open("foo.pkl", 'wb') as f:
#     pickle.dump(embeddings, f)
# with open("foo.pkl", 'rb') as f:
#     new_docsearch = pickle.load(f)


import pickle
with open("foo.pkl", 'rb') as f:
    new_docsearch = pickle.load(f)

with open('texts.pkl', 'rb') as f:
    texts = pickle.load(f)

docsearch = FAISS.from_texts(texts, new_docsearch)

#query = input("Please ask any question:" )

query = st.text_input("Please ask any question on OS:" )

docs = docsearch.similarity_search(query)

context = docs[0].page_content

#print(docs[0].page_content)

# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
# chain.run(input_documents=docs, question=query)

# Generate answer using OpenAI API
response = openai.Completion.create(
        #engine="gpt-3.5-turbo",
        engine="text-davinci-003",
        prompt=f"I am a Q&A bot. I answer the following question {query} based on the following context:{context}",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
# Extract the answer from the response
answer = response.choices[0].text.strip()
 
st.text(answer)
 
#print (answer)