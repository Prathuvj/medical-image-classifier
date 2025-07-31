import os
from PIL import Image

def count_images_per_class(data_dir):
    stats = {}
    for folder in os.listdir(data_dir):
        path = os.path.join(data_dir, folder)
        if os.path.isdir(path):
            stats[folder] = len([f for f in os.listdir(path) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
    return stats