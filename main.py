from fastapi import FastAPI, File, UploadFile
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import function1
import function2
import Function3
import function4
import Function5

app = FastAPI()


url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
output_file = "images.tar"
function1.main()
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
path = "images/Images"
dataset = function2.DogDataset(path,one_hot=0)
image_paths = dataset.X
ids = dataset.y
embeddings_with_ids = Function3.embed_images_with_ids(image_paths, ids, batch_size=64, model=model, processor=processor, device=device)
img_id, embedding = embeddings_with_ids[0]
dimension = embedding.shape[1]
index = function4.initialize_index(dimension, search_method="hnsw") #database
function4.add_embeddings_to_index(index, embeddings_with_ids)
    
@app.post("/search/")

async def search_image(file: UploadFile = File(...), top_k: int = 5):
    # Read image from frontend
    image = Image.open(file.file)
    # Transfer to vector_embedding by using CLIP
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        query_vector = model.get_image_features(**inputs).cpu().numpy().flatten()

    results, distances = Function5.retrieve_closest_vectors(index, query_vector, top_k=top_k)
    
    similar_images = []
    for result in results:
        image_path = os.path.join("images/Images", f"{result}.jpg")
        similar_images.append({"image_path": image_path, "distance": distances[result]})


    return {"results": similar_images}






    





    