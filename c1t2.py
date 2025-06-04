import os

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ====== 설정 ======
DATASET_DIR = "Large"
IMAGE_SIZE = (128, 128)
HOG_PIXELS_PER_CELL = (32, 32)
K = 1  # Task 2 Retrieval용 Top-10
print("IMAGE_SIZE:", IMAGE_SIZE, "HOG_PIXELS_PER_CELL:", HOG_PIXELS_PER_CELL, " K : ", K)

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


# ====== 3. 데이터 로딩 ======
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

# ====== 8. KNN 훈련 (Task 2용 - Retrieval) ======
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train_hog, y_train_enc)

# ====== 9. Top-10 Retrieval 평가 ======
distances, indices = knn.kneighbors(X_val_hog, n_neighbors=K)

correct_count = 0
all_correct_flags = []
top10_labels_list = []

for i in range(len(indices)):
    top10_label_indices = y_train_enc[indices[i]]  # Top-10 인덱스 → 숫자 라벨
    top10_label_names = le.inverse_transform(top10_label_indices)  # → 문자열 라벨
    top10_label_names = list(top10_label_names)  # numpy array → list 변환
    top10_labels_list.append(','.join(top10_label_names))  # 쉼표로 연결된 문자열

    is_all_same = np.all(top10_label_indices == y_val_enc[i])
    all_correct_flags.append(is_all_same)
    if is_all_same:
        correct_count += 1

top10_accuracy = correct_count / len(indices)
print(f"Task 2 - Validation Top-10 Retrieval Accuracy: {top10_accuracy * 100:.2f}%")

# ====== 10. CSV 저장 (Top-10 라벨 포함) ======
df = pd.DataFrame({
    "TrueLabel": y_val,
    "Top10Labels": top10_labels_list,
    "AllTop10Correct": all_correct_flags
})
df.to_csv("challenge1_task2_val_retrieval_results.csv", index=False)

print("결과 저장 완료:")
print("- challenge1_task2_val_retrieval_results.csv")
