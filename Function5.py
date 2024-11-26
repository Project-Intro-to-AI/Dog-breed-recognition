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
