import json
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ====== 설정 ======
TEST_DIR = "test"
FEATURE_DB_PATH = "X_train.npy"
LABEL_DB_PATH = "y_train.npy"
CLASS_MAP_PATH = "class_mapping.json"
MODEL_PATH = "resnet18_feature_extractor.pt"
OUTPUT_CSV = "c2_t2_a1.csv"
K = 10
BATCH_SIZE = 32
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== transform 정의 ======
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ====== Custom Dataset (단일 디렉토리에서 불러오기) ======
class TestImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name


# ====== 데이터 로딩 ======
test_dataset = TestImageDataset(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ====== 모델 및 feature DB 불러오기 ======
model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = model.to(DEVICE)
model.eval()

X_train = np.load(FEATURE_DB_PATH)
y_train = np.load(LABEL_DB_PATH)

with open(CLASS_MAP_PATH, "r") as f:
    idx_to_class = json.load(f)

# ====== test feature 추출 ======
X_test = []
filenames = []
with torch.no_grad():
    for imgs, fnames in tqdm(test_loader, desc="Extracting test features"):
        imgs = imgs.to(DEVICE)
        feats = model(imgs).cpu().numpy()
        X_test.append(feats)
        filenames.extend(fnames)

X_test = np.vstack(X_test)

# ====== Nearest Neighbors 기반 Top-K 검색 ======
knn = NearestNeighbors(n_neighbors=K)
knn.fit(X_train)
dists, indices = knn.kneighbors(X_test)

# ====== 결과 처리 및 저장 ======
rows = []
for i, idxs in enumerate(indices):
    topK_labels = [idx_to_class[str(y_train[idx])] for idx in idxs]
    row = [filenames[i]] + topK_labels
    rows.append(row)

submission = pd.DataFrame(rows, columns=["QueryImage"] + [f"Top{i + 1}" for i in range(K)])
submission.to_csv(OUTPUT_CSV, index=False)
print(f"✅ CSV 저장 완료: {OUTPUT_CSV}")
