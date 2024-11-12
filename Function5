import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_closest_vectors(query_vector, database_vectors, top_k=5):
    # Calculate cosine similarity between the query vector and database vectors
    similarities = cosine_similarity([query_vector], database_vectors)[0]
    print(similarities)
    
    # Get the indices of the top_k closest vectors
    closest_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return closest_indices.tolist()

# Example Usage:
# Let's assume `database_vectors` is a list of vectors stored in your database
# And `query_vector` is the vector you want to find the nearest neighbors for

database_vectors = np.random.rand(10, 128)  # Example database with 10 vectors of 128 dimensions
query_vector = np.random.rand(128)          # Example query vector of 128 dimensions

closest_ids = find_closest_vectors(query_vector, database_vectors)
print("IDs of the 5 closest vectors:", closest_ids)
