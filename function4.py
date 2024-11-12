#function 4
from typing import List, Tuple
import faiss  # FAISS cần được cài đặt cho lưu trữ và tìm kiếm vector
import numpy as np
import torch

def initialize_index(dimension: int, search_method: str = "flat"):
    if search_method == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW với 32 kết nối
    else:
        index = faiss.IndexFlatL2(dimension)  # Flat index với khoảng cách L2

    return index

def add_embeddings_to_index(index, embeddings_with_ids: List[Tuple[int, torch.Tensor]]):
    ids, embeddings = zip(*embeddings_with_ids)
    embeddings_np = np.vstack([embedding.numpy() for embedding in embeddings]).astype(np.float32)

    # Thêm embeddings vào index
    index.add(embeddings_np)

    # Kiểm tra số lượng embeddings đã thêm
    if index.ntotal == len(embeddings_with_ids):
        print("All embeddings added successfully.")
    else:
        print("Error: Some embeddings were not added.")


# Example usage
dimension = 512  # Đối với CLIP, kích thước của embedding thường là 512
index = initialize_index(dimension, search_method="hnsw")
add_embeddings_to_index(index, embeddings_with_ids)

print(f"Total embeddings in index: {index.ntotal}")

