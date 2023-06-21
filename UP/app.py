from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch

# Load the dataset
data = pd.read_csv('data.csv')
data = data.drop('code', axis=1)

# Preprocess the dataset
def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'[-+]?[0-9]+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()  # Remove whitespace
    return text

data['title'] = data['title'].apply(text_preprocessing)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode the data
data['text'] = data['title']
data['text'] = data['text'].apply(lambda x: x[:512])  # Limit the input length to 512 tokens

# Function to generate embeddings and cache the results
@lru_cache(maxsize=None)
def generate_embeddings():
    with torch.no_grad():
        encoded_inputs = tokenizer(data['text'].tolist(), padding='longest', truncation=True, max_length=512, return_tensors='pt')
        model_inputs = {k: v.to(model.device) for k, v in encoded_inputs.items()}
        outputs = model(**model_inputs)
        embeddings = outputs[0][:, 0, :].numpy()
    return embeddings

# Initialize the embeddings cache
embeddings_cache = generate_embeddings()

# Initialize the Flask app
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for search API
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']  # Get the query from the form
    search_results = perform_search(query)  # Perform the search
    return jsonify(search_results)

# Function to perform search
def perform_search(query, top_k=3):
    # Retrieve the cached embeddings
    embeddings = embeddings_cache

    # Encode the query
    encoded_query = tokenizer.encode_plus(query, padding='longest', truncation=True, max_length=512, return_tensors='pt')

    # Generate the query embedding
    with torch.no_grad():
        query_inputs = {k: v.to(model.device) for k, v in encoded_query.items()}
        query_outputs = model(**query_inputs)
        query_embedding = query_outputs[0][:, 0, :].numpy()

    # Calculate cosine similarity between query embedding and dataset embeddings
    similarities = cosine_similarity(query_embedding, embeddings)

    # Get the indices of top-k most similar documents
    top_indices = similarities.argsort()[0][-top_k:][::-1]

    # Retrieve the top-k documents
    results = data.loc[top_indices].to_dict(orient='records')
    return results

# Run the Flask app
if __name__ == '__main__':
    app.run()
