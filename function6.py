from collections import Counter

def get_most_common_breed(breed_list):
    """
    Xác định giống chó xuất hiện nhiều nhất trong danh sách.
    
    Args:
        breed_list (list of str): Danh sách tên giống chó từ 5 ảnh trả về.
        
    Returns:
        str: Tên giống chó xuất hiện nhiều nhất.
    """
    if not breed_list or not isinstance(breed_list, list):
        return "Invalid input: breed_list must be a non-empty list of strings."
    
    # Đếm tần suất xuất hiện của từng giống chó
    breed_counter = Counter(breed_list)
    
    # Tìm giống chó xuất hiện nhiều nhất
    most_common_breed, _ = breed_counter.most_common(1)[0]
    
    return most_common_breed
