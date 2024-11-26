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
if __name__ == '__main__':
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
    query_vector = np.random.randn(dimension).astype(np.float32)
    result, distance = Function5.retrieve_closest_vectors(index, query_vector, top_k=5)
    print(result)






    





    