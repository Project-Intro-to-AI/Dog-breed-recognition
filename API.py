# from fastapi import FastAPI, File, UploadFile
# from typing import List

# app = FastAPI()

# # Core của bạn - giả sử là hàm xử lý ảnh
# def find_similar_images(uploaded_image: bytes) -> List[str]:
#     # Đây là hàm bạn đã viết ở phần core
#     # Giả sử nó trả về danh sách tên ảnh giống nhất
#     return ["image1.jpg", "image2.jpg", "image3.jpg"]

# # API nhận file ảnh từ frontend
# @app.post("/upload/")
# async def upload_image(file: UploadFile = File(...)):
#     image_data = await file.read()  # Đọc file
#     similar_images = find_similar_images(image_data)  # Gọi core để xử lý
#     return {"similar_images": similar_images}
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)  