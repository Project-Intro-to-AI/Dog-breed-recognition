from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import function1  # Function to download data set
from function2 import DogDataset  # Function to load data set
import Function3  # Function to embed image
import function4  # Function to add embeddings to database
import Function5  # Function to retrieve vectors from database
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from collections import Counter
from utils import BATCH, IMG_PATH, MODEL_NAME, BASE_URL, cosine
from fastapi.middleware.cors import CORSMiddleware


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize FastAPI app and mount static files
app = FastAPI()
app.mount("/images", StaticFiles(directory=r"images/"), name="images")
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>Call API Example</title>
        </head>
        <body>
            <h1>Call API Example</h1>
            <a href="/call_api">Call API</a>
        </body>
    </html>
    """

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các domain
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả các headers
)


@app.get("/call_api")
def call_api():
    # Here, you would call the external API
    return {"message": "API called successfully!"}
# Preprocess: Download dataset, initialize CLIP model, and prepare embeddings
function1.main()
# load model
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# Load dataset and embeddings
dataset = DogDataset(IMG_PATH)
image_paths = dataset.X 
ids = dataset.y 
embeddings_with_ids = Function3.main(image_paths, ids)
# print (embeddings_with_ids)
# exit(0)

# print(dataset.X[embeddings_with_ids[0][0]],embeddings_with_ids[0][1])
# print(f"cosine similarity {cosine(embeddings_with_ids[0][1], embeddings_with_ids[4636][1])}")
# exit(0)
def KNN(feat,top_k=5):
    similarities = [(i, cosine(embeddings_with_ids[i][1],feat)) for i in range(len(embeddings_with_ids))]
    similarities = sorted(similarities,key=lambda x:x[1], reverse=True)[:top_k]
    top_labels = [dataset.y[x[0]] for x in similarities]
    most_common_label,_ = Counter(top_labels).most_common(1)[0]
    return most_common_label, similarities

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
            # query_vector = np.array(query_vector) / np.linalg.norm(query_vector)
        print(f"query_vector:{query_vector}, {cosine(embeddings_with_ids[0][1],query_vector):.4f}")

        most_common_labels,super_result = KNN(query_vector)
        print(super_result)
        # Filter for unique results
        # Prepare response with image paths, distances, and breed names
        similar_images = []
        for result, distance in super_result:
            relative_path = os.path.relpath(dataset.X[result], "images")  
            # print(relative_path)# Get relative path
            image_path = f"{BASE_URL}/images/{relative_path.replace(os.sep, '/')}"  # Construct absolute URL
            
            # Map class ID to breed name using dataset.class_names
            class_id = dataset.y[result].item()  # Get class ID for the image
            breed_name = dataset.class_names[class_id] if class_id < len(dataset.class_names) else "Unknown"

            similar_images.append({
                "image_path": image_path,
                "distance": float(distance),
                "breed_name": breed_name
            })

        return {"results": similar_images}

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Log detailed error to console
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
