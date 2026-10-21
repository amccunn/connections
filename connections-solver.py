import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from itertools import combinations

load_dotenv()

#get the words for the connections
connections_words = ["ENGLISH HORN",
                    "LEAPFROG",
                    "ALPHA DOG",
                    "SWEETBREAD",
                    "HOT POTATO",
                    "MANTIS SHRIMP",
                    "SUNFLOWERS",
                    "BIG CHEESE",
                    "AMERICAN GOTHIC",
                    "GRAPE-NUTS",
                    "CAMPBELL'S SOUP CAN",
                    "DUCK DUCK GOOSE",
                    "TOP BANANA",
                    "THE SCREAM",
                    "MUSICAL CHAIRS",
                    "HEAVYWEIGHT"]

#load the transformer model and get the embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
word_embeddings = model.encode(connections_words, convert_to_tensor=True)

def score_group(group_indices):
  
  group_vecs = word_embeddings[list(group_indices)]

  #calculate pairwise similarities within the 4 words
  sim_matrix = util.cos_sim(group_vecs, group_vecs)

  #dont compare them to themselves
  mask = ~np.eye(4, dtype=bool)

  return sim_matrix[mask].mean().item()

all_combinations = list(combinations(range(16), 4))
scored_groups = []

for combo in all_combinations:
  score = score_group(combo)
  scored_groups.append((score, combo))

#sort by highest likely hood of group
scored_groups.sort(key=lambda x: x[0], reverse=True)

#print top 5 scores
print("Top predicted categories:")

for score, combo in scored_groups[:5]:
  
  group_words = [connections_words[i] for i in combo]

  print(f"Score: {score:.3f} -> {group_words}")
