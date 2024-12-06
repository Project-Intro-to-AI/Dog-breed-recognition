import requests
import tarfile
import os
import hashlib

# URL to the dataset
url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
output_file = "images.tar"

def compute_md5(file_path): # tính "dấu vân tay" của file
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()

# Download the file with progress display
def download_dataset(): # ghi toàn bộ dữ liệu từ URL của dataset vào file
    print("Downloading dataset...")
    response = requests.get(url, stream=True) # send GET request to URL, stream = True : tải dữ liệu theo từng đoạn
    total_size = int(response.headers.get('Content-Length', 0))  # Total file size in bytes
    block_size = 1024  # Chunk size
    progress = 0

    with open(output_file, "wb") as file:
        for data in response.iter_content(block_size):
            progress += len(data)
            file.write(data) # write data into output_file
            percent = (progress / total_size) * 100
            print(f"\rProgress: {percent:.2f}%", end="")  # Print progress in the same line

    print("\nDownload complete.")

def verify(): # check md5
    expected_md5 = "1bb1f2a596ae7057f99d7d75860002ef" 
    downloaded_md5 = compute_md5(output_file)
    if downloaded_md5 == expected_md5:
        print("File integrity verified.")
    else:
        print(f"Warning: File integrity check failed! (MD5: {downloaded_md5})")

# Extract the tar file
def extract():
    print("Extracting dataset...")
    if tarfile.is_tarfile(output_file): # check xem có phải là filetar hay không
        with tarfile.open(output_file) as tar:
            tar.extractall(path="./images")  # Extract to 'images' directory
            print("Extraction complete.")
    else:
        print("The downloaded file is not a valid tar file.")


def main():
    if os.path.exists("images.tar"): print("File is already downloaded.")
    else: 
        download_dataset()
        verify()
        extract()
if __name__ == '__main__':
    main()