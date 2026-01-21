import numpy as np
import os
import gzip
import urllib.request

def mnist():
    url_base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    file_names = {
        "x_train": "train-images-idx3-ubyte.gz",
        "y_train": "train-labels-idx1-ubyte.gz",
        "x_test": "t10k-images-idx3-ubyte.gz",
        "y_test": "t10k-labels-idx1-ubyte.gz"
    }
    
    data = {}
    save_dir = "./data"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
        
    for key, filename in file_names.items():
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(url_base + filename, filepath)
            
        with gzip.open(filepath, 'rb') as f:
            if "images" in filename:
                # Skip 16-byte header
                arr = np.frombuffer(f.read(), np.uint8, offset=16)
                data[key] = arr.reshape(-1, 28, 28).astype(np.float32) / 255.0
            else:
                # Skip 8-byte header
                data[key] = np.frombuffer(f.read(), np.uint8, offset=8)
                
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]

# if __name__ == "__main__":
#     xt, yt, xv, yv = mnist()
#     print(f"X Train: {xt.shape}")  # Should be (60000, 28, 28)
#     print(f"Y Train: {yt.shape}")  # Should be (60000,)
