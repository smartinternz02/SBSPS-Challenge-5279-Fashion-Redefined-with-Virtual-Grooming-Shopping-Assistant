from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe = FeatureExtractor()

    for img_path in sorted(Path(r"C:\Users\WELCOME\AppData\Local\Programs\Python\Python38\Scripts\IBM Fashion guider project\static\img").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path(r"C:\Users\WELCOME\AppData\Local\Programs\Python\Python38\Scripts\IBM Fashion guider project\static\feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        print(feature_path)
        np.save(feature_path, feature)
