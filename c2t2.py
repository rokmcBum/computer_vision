import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# ====== 설정 ======
DATASET_DIR = "a"  # train/ & test/ 포함된 폴더
IMAGE_SIZE = 224
BATCH_SIZE = 32
K = 10  # Top-10 평가
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Transform 정의 ======
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ====== ResNet50 모델 정의 ======
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.fc = torch.nn.Identity()  # feature extractor
model = model.to(DEVICE)
model.eval()


# ====== Feature 추출 함수 ======
def extract_features(loader, with_path=False):
    features, labels, filenames = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features" + (" with path" if with_path else "")):
            if with_path:
                imgs, lbls, paths = batch
                filenames.extend([os.path.basename(p) for p in paths])
            else:
                imgs, lbls = batch

            imgs = imgs.to(DEVICE)
            feats = model(imgs).cpu().numpy()
            features.append(feats)
            labels.extend(lbls.numpy())

    results = [np.vstack(features), np.array(labels)]
    if with_path:
        results.append(filenames)
    return tuple(results)


# ====== Custom Dataset with Paths ======
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


# ====== 데이터 로딩 ======
train_dataset = ImageFolder(os.path.join(DATASET_DIR, "train"), transform=transform)
test_dataset = ImageFolderWithPaths(os.path.join(DATASET_DIR, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ====== Feature 추출 ======
X_train, y_train = extract_features(train_loader, with_path=False)
X_test, y_test, test_filenames = extract_features(test_loader, with_path=True)

# ====== KNN 학습 ======
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, y_train)

# ====== Top-10 예측 ======
distances, indices = knn.kneighbors(X_test, n_neighbors=K)
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

correct_count = 0
rows = []

for i, top_indices in enumerate(indices):
    top_labels_idx = y_train[top_indices]
    top_labels = [idx_to_class[j] for j in top_labels_idx]
    true_label = idx_to_class[y_test[i]]
    all_correct = all([lbl == true_label for lbl in top_labels])
    if all_correct:
        correct_count += 1
    row = [f"query{i + 1:03}.png", true_label] + top_labels
    rows.append(row)

accuracy = correct_count / len(X_test)
print(f"\n✅ Task 2 - Top-10 Retrieval Accuracy: {accuracy * 100:.2f}%")

# ====== CSV 저장 ======
columns = ["QueryImage", "TrueLabel"] + [f"Top{i + 1}" for i in range(K)]
df = pd.DataFrame(rows, columns=columns)
df.to_csv("challenge2_task2_submission.csv", index=False)
print("📄 CSV 저장 완료: challenge2_task2_submission.csv")
