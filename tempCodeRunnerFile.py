List of corresponding IDs
# embeddings_with_ids = Function3.main(image_paths, ids)

# # dimension = 512  # Get embedding dimension
# index = function4.initialize_index(512, search_method = "flat", use_gpu = False)
# index = function4.add_embeddings_to_index(index, embeddings_with_ids,index_file = "faiss_index.bin", use_gpu = False)
# print("Number of vectors in FAISS index:", index.ntotal)


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
#             # image_filename = f"{result}.jpg"
#             # image_path = f"/images/Images/{image_filename}"
#             similar_images.append({"image_path": dataset.X[result], "distance": distance})

        
#         print(similar_images)
#         return {"results": similar_images}
        

#     except Exception as e:
#         print(f"Error occurred: {str(e)}")  # Ghi log lỗi chi tiết vào console
#         raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


