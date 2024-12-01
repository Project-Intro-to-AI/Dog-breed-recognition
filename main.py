from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import function1  # Function to download data set
import function2  # Function to load data set
import Function3  # Function to embed image
import function4  # Function to add embeddings to database
import Function5  # Function to retrieve vectors from database
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app and mount static files
app = FastAPI()
app.mount("/images", StaticFiles(directory="images"), name="images")

# Preprocess: Download dataset, initialize CLIP model, and prepare embeddings
url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
output_file = "images.tar"
function1.main()  # Download dataset
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load dataset and embeddings
path = "images/Images"
dataset = function2.DogDataset(path, one_hot=0)
image_paths = dataset.X  # List of image file paths
ids = dataset.y  # List of corresponding IDs
embeddings_with_ids = Function3.embed_images_with_ids(
    image_paths, ids, batch_size=64, model=model, processor=processor, device=device
)
dimension = 512  # Get embedding dimension
index = function4.initialize_index(dimension, search_method="hnsw")
function4.add_embeddings_to_index(index, embeddings_with_ids)


@app.post("/search/")
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    """
    Endpoint to search for similar images based on the uploaded image.
    """
    try:
        # Validate the uploaded file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

        # Read and process the image
        image = Image.open(file.file).convert("RGB")  # Ensure RGB mode
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Get query vector (embedding) for the uploaded image
        with torch.no_grad():
            query_vector = model.get_image_features(**inputs).cpu().numpy().flatten()

        # Retrieve similar images
        results, distances = Function5.retrieve_closest_vectors(index, query_vector, top_k=top_k)

        # Prepare response with image paths and distances
        similar_images = []
        for result, distance in zip(results, distances):
            image_filename = f"{result}.jpg"
            image_path = f"/images/Images/{image_filename}"
            similar_images.append({"image_path": image_path, "distance": distance})

        return {"results": similar_images}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))