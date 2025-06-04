import os

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ====== 설정 ======
DATASET_DIR = "Large"
IMAGE_SIZE = (128, 128)  # HOG는 64×64 기준이 안정적
HOG_PIXELS_PER_CELL = (32, 32)
K = 9  # KNN 이웃 수
print("IMAGE_SIZE: ", IMAGE_SIZE, "HOG_PIXELS_PER_CELL: ", HOG_PIXELS_PER_CELL, "K: ", K)
CLASS_NAMES = [
    'Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney',
    'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light'
]


# ====== 1. 이미지 로딩 함수 ======
def load_images(folder_path, valid_classes, size=(64, 64)):
    images, labels = [], []
    for class_name in os.listdir(folder_path):
        if class_name not in valid_classes:
            continue
        class_path = os.path.join(folder_path, class_name)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, size)
            images.append(img)
            labels.append(class_name)
    return np.array(images), np.array(labels)


# ====== 2. HOG 피처 추출 함수 ======
def extract_hog_features(images, pixels_per_cell=(8, 8)):
    features = []
    for img in tqdm(images, desc="Extracting HOG"):
        hog_feature = hog(
            img, orientations=9,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features.append(hog_feature)
    return np.array(features)


# ====== 3. 데이터 불러오기 ======
X_all_img, y_all = load_images(DATASET_DIR, CLASS_NAMES, size=IMAGE_SIZE)
print("총 이미지 개수:", len(X_all_img))

# ====== 4. Train/Validation Split ======
X_train_img, X_val_img, y_train, y_val = train_test_split(
    X_all_img, y_all, test_size=0.1, stratify=y_all, random_state=42
)
print(f"Train: {len(X_train_img)} / Validation: {len(X_val_img)}")

# ====== 5. HOG 피처 추출 ======
X_train_hog = extract_hog_features(X_train_img, pixels_per_cell=HOG_PIXELS_PER_CELL)
X_val_hog = extract_hog_features(X_val_img, pixels_per_cell=HOG_PIXELS_PER_CELL)
print("HOG feature shape:", X_train_hog.shape)

# ====== 6. 라벨 인코딩 ======
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

# ====== 7. KNN 훈련 ======
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train_hog, y_train_enc)

# ====== 8. 예측 및 평가 ======
y_val_pred = knn.predict(X_val_hog)
y_val_pred_labels = le.inverse_transform(y_val_pred)

accuracy = accuracy_score(y_val_enc, y_val_pred)
print(f"Validation Top-1 Accuracy: {accuracy * 100:.2f}%")

# ====== 9. 결과 저장 ======
val_results = pd.DataFrame({
    "TrueLabel": y_val,
    "PredictedLabel": y_val_pred_labels
})
val_results.to_csv("challenge1_task1_val_predictions_HOG.csv", index=False)
print("결과 저장 완료: challenge1_task1_val_predictions_HOG.csv")
