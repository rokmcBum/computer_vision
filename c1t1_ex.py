import csv
import os

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ====== 설정 ======
TRAIN_DIR = "Large"
TEST_DIR = "test"  # test/classname/pic.png → test/pic.png
IMAGE_SIZE = (128, 128)
HOG_PIXELS_PER_CELL = (32, 32)
K = 9

CLASS_NAMES = [
    'Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney',
    'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light'
]

print("IMAGE_SIZE:", IMAGE_SIZE, "HOG:", HOG_PIXELS_PER_CELL, "K:", K)


# ====== 1. 학습용 이미지 로딩 (라벨 있음) ======
def load_train_images(folder_path, valid_classes, size=(64, 64)):
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


# ====== 2. 테스트 이미지 로딩 (라벨 없음) ======
def load_test_images(folder_path, size=(64, 64)):
    images, filenames = [], []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, size)
        images.append(img)
        filenames.append(filename)
    return np.array(images), filenames


# ====== 3. HOG 피처 추출 ======
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


# ====== 4. 학습 이미지 처리 ======
X_train_img, y_train = load_train_images(TRAIN_DIR, CLASS_NAMES, size=IMAGE_SIZE)
print("Train 이미지 수:", len(X_train_img))

# ====== 5. 테스트 이미지 처리 ======
X_test_img, test_filenames = load_test_images(TEST_DIR, size=IMAGE_SIZE)
print("Test 이미지 수:", len(X_test_img))

# ====== 6. 피처 추출 ======
X_train_feat = extract_hog_features(X_train_img, pixels_per_cell=HOG_PIXELS_PER_CELL)
X_test_feat = extract_hog_features(X_test_img, pixels_per_cell=HOG_PIXELS_PER_CELL)

# ====== 7. 라벨 인코딩 ======
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# ====== 8. KNN 훈련 ======
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train_feat, y_train_enc)

# ====== 9. 테스트 예측 ======
y_test_pred = knn.predict(X_test_feat)
y_test_pred_labels = le.inverse_transform(y_test_pred)

# ====== 10. 결과 저장 ======
with open('c1_t1_a1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for filename, label in zip(test_filenames, y_test_pred_labels):
        writer.writerow([filename, label])

print("✅ CSV 저장 완료: c1_t1_a1.csv")
