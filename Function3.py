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

def embed_images_with_ids(image_paths, ids, batch_size=64, model=model, processor=processor, device=device):
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

        # Generate embeddings
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs).cpu()
        del inputs
        torch.cuda.empty_cache() # Làm rỗng cache sau mỗi batch

        # Add embeddings with corresponding IDs to the result list
        for img_id, embedding in zip(batch_ids, embeddings):
            embeddings_with_ids.append((img_id, embedding))

    return embeddings_with_ids

# # Example usage
# path = "images/Images"
# dataset = function2.DogDataset(path, one_hot=0)
# image_paths = dataset.X
# ids = dataset.y

# embeddings_with_ids = embed_images_with_ids(image_paths, ids)
# for img_id, embedding in embeddings_with_ids:
#     print(f"ID: {img_id}, Embedding: {embedding}")