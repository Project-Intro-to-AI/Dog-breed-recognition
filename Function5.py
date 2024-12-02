import numpy as np
import faiss

def retrieve_closest_vectors(index, query_vector: np.ndarray, top_k: int = 5):
    """
    Tìm kiếm top-k vector gần nhất trong index FAISS.

    Args:
        index: FAISS index đã được khởi tạo và thêm vector.
        query_vector: Vector truy vấn (embedding) với kích thước (1, dimension).
        top_k: Số lượng vector gần nhất cần tìm (default = 5).

    Returns:
        List chứa ID của top-k vector gần nhất trong index.
    """
    # Đảm bảo query_vector có đúng định dạng numpy array (dạng float32)
    if not isinstance(query_vector, np.ndarray):
        query_vector = np.array(query_vector, dtype=np.float32)
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    # Tìm kiếm trong FAISS
    distances, indices = index.search(query_vector, top_k)

    
    
    # Trả về danh sách các ID của vector gần nhất
    return indices[0].tolist(), distances[0].tolist()
# Tạo FAISS index
dimension = 512  # Kích thước vector
index = faiss.IndexFlatL2(dimension)

# Thêm một số vector vào index
np.random.seed(42)  # Đặt seed để kết quả nhất quán
vectors = np.random.random((10, dimension)).astype(np.float32)
index.add(vectors)

# Tạo query vector
query_vector = np.random.random((1, dimension)).astype(np.float32)

# Gọi hàm retrieve_closest_vectors
indices, distances = retrieve_closest_vectors(index, query_vector, top_k=3)

# In kết quả
print("Query Vector Shape:", query_vector.shape)
print("Retrieved Indices:", indices)
print("Distances:", distances)