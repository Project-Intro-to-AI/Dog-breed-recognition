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
from fastapi.responses import HTMLResponse
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
from fastapi.middleware.cors import CORSMiddleware

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
url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
output_file = "images.tar"
function1.main()  # Download dataset
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load dataset and embeddings
path = r"images\Images"
dataset = function2.DogDataset(path)
image_paths = dataset.X  # List of image file paths
# print(image_paths[1])
ids = dataset.y  # List of corresponding IDs
embeddings_with_ids = Function3.main(image_paths, ids)

# dimension = 512  # Get embedding dimension
index = function4.initialize_index(512, search_method = "flat")
index = function4.add_embeddings_to_index(index, embeddings_with_ids,index_file = "faiss_index.bin")
print("Number of vectors in FAISS index:", index.ntotal)


# @app.post("/search/")
# async def search_image(file: UploadFile = File(...), top_k: int = 5):
#     """
#     Endpoint to search for similar images based on the uploaded image.
#     """
#     try:
#         # Validate the uploaded file
#         if not file.content_type.startswith("image/"):
#             raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

#         # Read and process the image
#         image = Image.open(file.file).convert("RGB")  # Ensure RGB mode
#         inputs = processor(images=image, return_tensors="pt").to(device)

#         # Get query vector (embedding) for the uploaded image
#         with torch.no_grad():
#             query_vector = model.get_image_features(**inputs).cpu().numpy().flatten()

#         # Retrieve similar images
#         results, distances = Function5.retrieve_closest_vectors(index, query_vector, top_k=top_k)

#         # Prepare response with image paths and distances
#         similar_images = []
#         for result, distance in zip(results, distances):
#             relative_path = os.path.relpath(dataset.X[result], "images")  # Lấy đường dẫn tương đối
#             image_path = f"D:/PROJECTINTROTOAI/images/{relative_path.replace(os.sep, '/')}"  # Định dạng đường dẫn hợp lệ cho web
#             similar_images.append({"image_path": image_path, "distance": distance})

        
#         print(similar_images)
#         return {"results": similar_images}
        

#     except Exception as e:
#         print(f"Error occurred: {str(e)}")  # Ghi log lỗi chi tiết vào console
#         raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
BASE_URL = "http://127.0.0.1:8000"  # Base URL of your server

@app.post("/search/")
# async def search_image(file: UploadFile = File(...), top_k: int = 5):
#     """
#     Endpoint to search for similar images based on the uploaded image.
#     """
#     try:
#         # Validate the uploaded file
#         if not file.content_type.startswith("image/"):
#             raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

#         # Read and process the image
#         image = Image.open(file.file).convert("RGB")  # Ensure RGB mode
#         inputs = processor(images=image, return_tensors="pt").to(device)

#         # Get query vector (embedding) for the uploaded image
#         with torch.no_grad():
#             query_vector = model.get_image_features(**inputs).cpu().numpy().flatten()

#         # Retrieve similar images
#         results, distances = Function5.retrieve_closest_vectors(index, query_vector, top_k=top_k)
#         print(f"result:{results}")
#         # Prepare response with web-accessible image paths and distances
#         similar_images = []
#         for result, distance in zip(results, distances):
#             relative_path = os.path.relpath(dataset.X[result], "images")  # Get relative path
#             image_path = f"{BASE_URL}/images/{relative_path.replace(os.sep, '/')}"  # Construct absolute URL
#             similar_images.append({"image_path": image_path, "distance": distance})
#         print(similar_images)
#         return {"results": similar_images}
#     except Exception as e:
#         print(f"Error occurred: {str(e)}")  # Log detailed error to console
#         raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


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
        results, distances = Function5.retrieve_closest_vectors(index, query_vector, top_k=top_k * 2)  # Fetch more to ensure uniqueness

        # Filter for unique results
        unique_results = []
        unique_indices = set()
        for result, distance in zip(results, distances):
            if result not in unique_indices:
                unique_indices.add(result)
                unique_results.append((result, distance))
                if len(unique_results) == top_k:
                    break

        # Prepare response with image paths, distances, and breed names
        similar_images = []
        for result, distance in unique_results:
            relative_path = os.path.relpath(dataset.X[result], "images")  # Get relative path
            image_path = f"{BASE_URL}/images/{relative_path.replace(os.sep, '/')}"  # Construct absolute URL
            
            # Map class ID to breed name using dataset.class_names
            class_id = dataset.y[result].item()  # Get class ID for the image
            breed_name = dataset.class_names[class_id] if class_id < len(dataset.class_names) else "Unknown"

            similar_images.append({
                "image_path": image_path,
                "distance": distance,
                "breed_name": breed_name
            })

        return {"results": similar_images}

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Log detailed error to console
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
