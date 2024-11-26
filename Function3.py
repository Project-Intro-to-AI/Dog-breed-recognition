import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def embed_images_with_ids(image_paths, ids):
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Prepare the inputs
    images = [Image.open(image_path) for image_path in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True)

    # Generate embeddings
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

    # Return a list of embeddings with corresponding IDs
    return list(zip(ids, embeddings))

# Example usage:
image_paths = ["image1.jpg"]
ids = [0]
embeddings_with_ids = embed_images_with_ids(image_paths, ids)

# Print the results
for img_id, embedding in embeddings_with_ids:
    print(f"ID: {img_id}, Embedding: {embedding}")
