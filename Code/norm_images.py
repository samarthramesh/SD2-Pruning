import sys
import os

from PIL import Image

def main(path):
    norm_path = path+"_norm"
    if os.path.isdir(norm_path):
        print("Normalised dataset already exists")
        return
    
    os.mkdir(norm_path)
    for file_path in os.listdir(path):
        image_path = path + "/" + file_path
        image = Image.open(image_path).convert("RGB").resize((299, 299), resample=Image.BICUBIC)
        image.save(norm_path + "/" + file_path)


if __name__ == "__main__":
    path = sys.argv[1]

    main(path)