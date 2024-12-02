import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
# import function2
import os

# Load model và processor một lần
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(device)

def embed_images_with_ids(image_paths, ids, batch_size=64):

    embeddings_with_ids = []

    # Kiểm tra thư mục ảnh
    if not os.path.exists("images/Images"):
        print(f"Error: Directory 'images/Images' does not exist.")
        return embeddings_with_ids

    # Process images in batches to save memory
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        # Kiểm tra sự tồn tại của file ảnh
        images = []
        for image_path in batch_paths:
            if not os.path.exists(image_path):
                print(f"Error: Image file '{image_path}' does not exist.")
                # Có thể bỏ qua ảnh lỗi hoặc dừng chương trình tùy theo yêu cầu
                # continue  # Bỏ qua ảnh lỗi và tiếp tục
                return embeddings_with_ids  # Dừng chương trình nếu có ảnh không tồn tại
            try:
                images.append(Image.open(image_path).convert("RGB")) # Thêm convert("RGB") để tránh lỗi về mode ảnh
            except Exception as e:
                print(f"Error loading image '{image_path}': {e}")
                # Có thể bỏ qua ảnh lỗi hoặc dừng chương trình tùy theo yêu cầu
                # continue  # Bỏ qua ảnh lỗi và tiếp tục
                # return embeddings_with_ids  # Dừng chương trình nếu có ảnh lỗi
        if not images:
            print(f"Warning: No images loaded in this batch. Continuing...")
            continue
        
        # Load and preprocess images
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        # print(device)

        # Generate embeddings
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs).cpu()
        del inputs
        torch.cuda.empty_cache() # Làm rỗng cache sau mỗi batch

        # Add embeddings with corresponding IDs to the result list
        for img_id, embedding in zip(batch_ids, embeddings):
            embeddings_with_ids.append((img_id, embedding))

    return embeddings_with_ids



import csv
import numpy as np

def save_embeddings_to_csv(file_path: str, embeddings_with_ids):
    """
    Lưu danh sách embeddings và IDs vào file CSV.
    """
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Ghi từng dòng ID và embedding
        for img_id, embedding in embeddings_with_ids:
            row = [int(img_id)] + embedding.cpu().numpy().tolist()
            writer.writerow(row)
    print(f"Embeddings saved to {file_path}")



def load_embeddings_from_csv(file_path: str):
    """
    Tải danh sách embeddings và IDs từ file CSV.
    """
    embeddings_with_ids = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        
        for row in reader:
            try:
                img_id = int(row[0])  # Cột đầu là ID
                embedding = np.array(row[1:], dtype=np.float32)  # Các cột còn lại là embedding
                embeddings_with_ids.append((img_id, torch.tensor(embedding)))
            except ValueError as e:
                print(f"Error processing row {row}: {e}")
    print(f"Embeddings loaded from {file_path}")
    return embeddings_with_ids


import os

def main(image_paths, ids, csv_file="embeddings.csv"):
    """
    Main function để kiểm tra file CSV và xử lý embedding.
    
    Args:
        image_paths (List[str]): Danh sách đường dẫn đến ảnh.
        ids (List[int]): Danh sách ID tương ứng với ảnh.
        batch_size (int): Kích thước batch để tạo embeddings.
        model: Mô hình dùng để tạo embeddings.
        processor: Bộ xử lý dữ liệu trước khi đưa vào model.
        device: Thiết bị tính toán ("cpu" hoặc "cuda").
        csv_file (str): Đường dẫn đến file CSV lưu embeddings (default = "embeddings.csv").
    """
    if not os.path.exists(csv_file):
        print(f"File {csv_file} không tồn tại. Tiến hành tạo embeddings...")
        # Tạo embeddings
        embeddings_with_ids = embed_images_with_ids(
            image_paths=image_paths,
            ids=ids,
        )
        # Lưu embeddings vào CSV
        save_embeddings_to_csv(csv_file, embeddings_with_ids)
    else:
        print(f"File {csv_file} đã tồn tại. Tiến hành tải embeddings...")
        # Tải embeddings từ CSV
        embeddings_with_ids = load_embeddings_from_csv(csv_file)
    
    print("Hoàn thành!")
    return embeddings_with_ids


# # Example usage
# path = "images/Images"
# dataset = function2.DogDataset(path, one_hot=0)
# image_paths = dataset.X
# ids = dataset.y

# embeddings_with_ids = embed_images_with_ids(image_paths, ids)
# for img_id, embedding in embeddings_with_ids:
#     print(f"ID: {img_id}, Embedding: {embedding}")


# if __name__ == "__main__":
#     image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]  # Thay bằng đường dẫn ảnh thực tế
#     ids = [1, 2, 3]
#     batch_size = 64
#     model = ...  # Khởi tạo model của bạn
#     processor = ...  # Khởi tạo processor
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     embeddings_with_ids = main(image_paths, ids, batch_size, model, processor, device)
#     print(f"Embeddings loaded or created: {len(embeddings_with_ids)}")
# image_paths = dataset.X  # List of image file paths
# ids = dataset.y
# embeddings_with_ids = main(image_paths, ids, csv_file="embeddings.csv")