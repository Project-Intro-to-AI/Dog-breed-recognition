import faiss
import numpy as np
from typing import List, Tuple
import torch
def initialize_index(dimension: int, search_method: str = "flat"):
    if search_method == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW với 32 kết nối
    else:
        index = faiss.IndexFlatL2(dimension)  # Flat index với khoảng cách L2
    return index


def save_index(index, file_path="faiss_index.bin"):
    """
    Lưu FAISS index vào file.

    Args:
        index: FAISS index đã khởi tạo.
        file_path (str): Đường dẫn file để lưu FAISS index.
    """
    print("Number of vectors in FAISSssssssssssssssssssssssss index:", index.ntotal)

    # Nếu index nằm trên GPU, chuyển về CPU
    if isinstance(index, faiss.IndexPreTransform) or hasattr(index, "to_cpu"):
        index = faiss.index_gpu_to_cpu(index)
    print("Number of vectors in FAISSssssssssssssssssssssssss index:", index.ntotal)

    # Lưu index
    faiss.write_index(index, file_path)
    print(f"Index saved to {file_path}")


def load_index(file_path: str, use_gpu: bool = False):
    """
    Tải FAISS index từ file.

    Args:
        file_path (str): Đường dẫn file chứa FAISS index.
        use_gpu (bool): Có chuyển index lên GPU hay không.

    Returns:
        index: FAISS index đã tải.
    """
    index = faiss.read_index(file_path)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    print(f"Index loaded from {file_path}")
    print("Number of vectors in FAISS index after loading:", index.ntotal)
    return index

def add_embeddings_to_index(index, embeddings_with_ids: List[Tuple[int, torch.Tensor]],index_file = "faiss_index.bin"):
    """
    Thêm embeddings cùng IDs vào FAISS index.

    Args:
        index: FAISS index đã khởi tạo (có hỗ trợ IDMap).
        embeddings_with_ids (List[Tuple[int, torch.Tensor]]): Danh sách (ID, embedding).

    Returns:
        None
    """
    import os

    # Kiểm tra file index tồn tại
    if os.path.exists(index_file):
        print(f"File {index_file} đã tồn tại. Tiến hành tải FAISS index...")
        index = load_index(index_file)
        print("Number of vectors in FAISS index after loadinguuuuuuuuuuuuuuuuuu:", index.ntotal)

        return index
    
    print(f"File {index_file} không tồn tại. Tiến hành tạo FAISS index...")

    # Tách IDs và embeddings
    ids, embeddings = zip(*embeddings_with_ids)
    
    # Chuyển embeddings từ PyTorch Tensor sang NumPy array
    embeddings_np = np.vstack([embedding.cpu().numpy() for embedding in embeddings]).astype(np.float32)
        
    # Thêm embeddings và IDs vào index
    index.add(embeddings_np)
    
    # Kiểm tra số lượng embeddings đã thêm
    print(f"Total embeddings in index: {index.ntotal}")
    if index.ntotal >= len(embeddings_with_ids):
        print("All embeddings added successfully.")
    else:
        print("Error: Some embeddings were not added.")
    
    # Lưu index vào file
    save_index(index, index_file)

    return index