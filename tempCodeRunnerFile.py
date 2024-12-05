# Import necessary libraries
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to download an image from a URL
def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content)).convert("RGB")  # Convert to RGB
        return image
    else:
        raise ValueError(f"Failed to download image from {image_url}")

# Function to embed an image using CLIP
def embed_image(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.cpu().numpy().flatten()

# URL of the image to download
image_url = "https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcQatZ8F2NcmsBWHDRyZJSdgNFIoYPyMn1_0OJvQ_kpija8ylyM1sug2s4cQm-LDajfTD7NqWaZ63peFlzfXTA99mA"  # Replace with your image URL

# Download and embed the image
try:
    print("Downloading image...")
    image = download_image(image_url)
    print("Image downloaded successfully.")

    print("Generating CLIP embedding...")
    embedding = embed_image(image)
    print("CLIP embedding generated.")

    # Print the embedding
    print("CLIP Embedding:")
    print(embedding)

except Exception as e:
    print(f"Error: {e}")
