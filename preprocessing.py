import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features = 10000
maxlen = 500

word_index = imdb.get_word_index()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def encode_text(text):
    text = clean_text(text)
    words = text.split()
    encoded = []
    for word in words:
        idx = word_index.get(word, 2)  
        if idx >= max_features:
            idx = 2
        encoded.append(idx)
    return encoded

def preprocess_for_model(text):
    encoded = encode_text(text)
    padded = pad_sequences([encoded], maxlen=maxlen)
    return padded
