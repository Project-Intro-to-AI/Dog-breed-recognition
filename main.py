from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import pandas as pd
from collections import Counter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import function1  # Function to download dataset
from function2 import DogDataset  # Function to load dataset
import Function3  # Function to embed image
from utils import cosine, BASE_URL, MODEL_NAME, IMG_PATH

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize FastAPI app
app = FastAPI()
app.mount("/images", StaticFiles(directory=r"images/"), name="images")

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>Dog Breed Identification</title>
        </head>
        <body>
            <h1>Dog Breed Identification API</h1>
            <p>Upload an image of a dog to identify its breed.</p>
        </body>
    </html>
    """

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load dog breed information from CSV
def load_dog_info(file_path):
    try:
        df = pd.read_csv(file_path, sep=';', skipinitialspace=True)
        df = df.dropna(subset=['Dog Breed'])  # Remove rows with missing 'Dog Breed'
        dog_info = {}
        for _, row in df.iterrows():
            if isinstance(row['Dog Breed'], str):  # Ensure value is a string
                breed = row['Dog Breed'].strip().lower().replace(" ", "_")
                dog_info[breed] = {
                    "origin": row.get('Origin', "N/A"),
                    "weight": row.get('Weight', "N/A").replace("�", "-"),
                    "height": row.get('Height', "N/A").replace("�", "-"),
                    "appearance": row.get('Apperance', "N/A"),
                    "lifespan": row.get('Lifespan', "N/A").replace("�", "-"),
                }
            else:
                print(f"Skipping invalid breed entry: {row['Dog Breed']}")
        return dog_info
    except Exception as e:
        print(f"Error loading dog info: {e}")
        return {}

# Load data
DOG_INFO_FILE = "Dog Information.csv"
dog_info = load_dog_info(DOG_INFO_FILE)

# Preprocess: Initialize dataset and embeddings
function1.main()
dataset = DogDataset(IMG_PATH)
image_paths = dataset.X
ids = dataset.y
embeddings_with_ids = Function3.main(image_paths, ids)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Define KNN function
def KNN(feat, top_k=5):
    similarities = [(i, cosine(embeddings_with_ids[i][1], feat)) for i in range(len(embeddings_with_ids))]
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    top_labels = [dataset.y[x[0]] for x in similarities]
    most_common_label, _ = Counter(top_labels).most_common(1)[0]
    return most_common_label, similarities

# Search Endpoint
@app.post("/search/")
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

        image = Image.open(file.file).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            query_vector = model.get_image_features(**inputs).cpu().numpy().flatten()

        most_common_label, super_result = KNN(query_vector, top_k=top_k)

        similar_images = []
        for result, distance in super_result:
            relative_path = os.path.relpath(dataset.X[result], "images")
            image_path = f"{BASE_URL}/images/{relative_path.replace(os.sep, '/')}"
            class_id = dataset.y[result].item()
            breed_name = dataset.class_names[class_id] if class_id < len(dataset.class_names) else "Unknown"

            breed_key = breed_name.strip().lower().replace(" ", "_")
            breed_info = dog_info.get(breed_key, {})

            similar_images.append({
                "image_path": image_path,
                "distance": float(distance),
                "breed_name": breed_name,
                "additional_info": breed_info
            })

        return {"results": similar_images}

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
