import numpy as np
BATCH = 64
IMG_PATH = r"images\Images"
MODEL_NAME = "openai/clip-vit-base-patch32"
BASE_URL = "http://127.0.0.1:8000"  # Base URL of your server

def cosine(a,b):
    a= np.array(a)
    b= np.array(b)
    return np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b) 