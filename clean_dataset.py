import os
from PIL import Image

DATASET_PATH = "E:\cnn_trained_dod\dataset"   # apna dataset path yahan do

def clean_folder(folder):
    removed = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()   # check image validity
            except:
                print("❌ Removing:", path)
                os.remove(path)
                removed += 1
    return removed

total_removed = clean_folder(DATASET_PATH)
print(f"\n✅ Cleaning done. Removed {total_removed} bad images.")
