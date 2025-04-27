import pandas as pd
import re
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv('webscrap.csv')

def clean_text(text):
    text = str(text)
    text = re.sub(r'must be 0 to purchase', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Restaurant'] = df['Restaurant'].apply(clean_text)
df['Section'] = df['Section'].apply(clean_text)
df['Item'] = df['Item'].apply(clean_text)
df['Description'] = df['Description'].apply(clean_text)

df['full_text'] = (
    "Restaurant: " + df['Restaurant'] + " | " +
    "Section: " + df['Section'] + " | " +
    "Item: " + df['Item'] + " | " +
    "Description: " + df['Description'] + " | " +
    "Price: " + df['Price']
)

print(df['full_text'].head())

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings = model.encode(df['full_text'].tolist(), convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

with open('csv_metadata.json', 'wb') as f:
    pickle.dump(df['full_text'].tolist(), f)

faiss.write_index(index, 'csv_vector_index.faiss')

print("✅ FAISS index and metadata saved successfully!")
