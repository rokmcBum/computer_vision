import json
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ====== 설정 ======
DATASET_DIR = "test"  # test 이미지 폴더
BATCH_SIZE = 32
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 1  # Challenge 2 Task 1은 Top-1 Accuracy

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
test_dataset = TestImageDataset(DATASET_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ====== 모델 및 저장된 feature 불러오기 ======
model = torch.load("resnet18_feature_extractor.pt", map_location=DEVICE, weights_only=False)
model = model.to(DEVICE)
model.eval()

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

with open("class_mapping.json", "r") as f:
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

# ====== KNN 분류기 학습 및 예측 ======
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred_labels = [idx_to_class[str(i)] for i in y_pred]

# ====== 결과 저장 ======
submission = pd.DataFrame({
    "Filename": filenames,
    "PredictedLabel": y_pred_labels
})
submission.to_csv("c2_t1_a1.csv", index=False)
print("✅ CSV 저장 완료: c2_t1_a1.csv")
