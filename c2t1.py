import os

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# ====== 설정 ======
DATASET_DIR = "a"  # train/ & test/ 포함된 폴더
IMAGE_SIZE = 256
BATCH_SIZE = 32
K = 1  # Top-1 평가용

# ====== transform 정의 ======
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop(224),  # ResNet50 권장
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ====== ResNet50 모델 정의 ======
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.fc = torch.nn.Identity()  # 마지막 FC 제거 → feature extractor
model.eval()


# ====== feature 추출 함수 ======
def extract_features(loader, model):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting features"):
            feats = model(imgs)
            features.append(feats.cpu().numpy())
            labels.extend(lbls.numpy())
    return np.vstack(features), np.array(labels)


# ====== 데이터 로딩 ======
train_dataset = ImageFolder(os.path.join(DATASET_DIR, "train"), transform=transform)
test_dataset = ImageFolder(os.path.join(DATASET_DIR, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ====== feature 추출 ======
X_train, y_train = extract_features(train_loader, model)
X_test, y_test = extract_features(test_loader, model)
print("Train feature shape:", X_train.shape)
print("Test feature shape:", X_test.shape)

# ====== KNN 학습 및 예측 ======
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# ====== 정확도 및 보고 ======
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Top-1 Accuracy on test set (ResNet50): {accuracy * 100:.2f}%")

# ====== 클래스 이름 매핑 ======
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
y_test_labels = [idx_to_class[i] for i in y_test]
y_pred_labels = [idx_to_class[i] for i in y_pred]

# ====== classification report 출력 ======
print("\n📊 Classification Report:")
print(classification_report(y_test_labels, y_pred_labels))
