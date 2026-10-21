import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from itertools import combinations

load_dotenv()

#get the words for the connections
connections_words = ["word",
                    "word",
                    "word",
                    "word",
                    "word",
                    "word",
                    "word",
                    "word",
                    "word",
                    "word",
                    "word",
                    "word",
                    "word",
                    "word",
                    "word",
                    "word",
                    "word"]

#load the transformer model and get the embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
word_embeddings = model.encode(connections_words, convert_to_tensor=True)

print(word_embeddings)


