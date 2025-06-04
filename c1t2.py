import csv
import os

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

# ====== 설정 ======
DATASET_DIR = "Large"
IMAGE_SIZE = (256, 256)
HOG_PIXELS_PER_CELL = (32, 32)
K = 10
print("IMAGE_SIZE:", IMAGE_SIZE, "HOG:", HOG_PIXELS_PER_CELL)

CLASS_NAMES = [
    'Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney',
    'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light'
]


# ====== 이미지 로딩 ======
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


# ====== Feature 추출 함수 (HOG + LBP + Histogram) ======
def extract_combined_features(images, pixels_per_cell=(8, 8)):
    features = []
    for img in tqdm(images, desc="Extracting Features"):
        # HOG
        hog_feature = hog(
            img, orientations=9,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )

        # LBP (Local Binary Pattern)
        lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
        (hist_lbp, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, 11),
                                     range=(0, 10))
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-7)

        # Grayscale Histogram (16 bins)
        hist_gray = cv2.calcHist([img], [0], None, [16], [0, 256])
        hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()

        # 결합
        combined = np.concatenate([hog_feature, hist_lbp, hist_gray])
        features.append(combined)
    return np.array(features)


# ====== 데이터 로딩 ======
X_all_img, y_all = load_images(DATASET_DIR, CLASS_NAMES, size=IMAGE_SIZE)
print("총 이미지 개수:", len(X_all_img))

# ====== Train/Test 분할 ======
X_train_img, X_val_img, y_train, y_val = train_test_split(
    X_all_img, y_all, test_size=0.1, stratify=y_all, random_state=42
)
print(f"Train: {len(X_train_img)} / Validation: {len(X_val_img)}")

# ====== Feature 추출 ======
X_train_feat = extract_combined_features(X_train_img, pixels_per_cell=HOG_PIXELS_PER_CELL)
X_val_feat = extract_combined_features(X_val_img, pixels_per_cell=HOG_PIXELS_PER_CELL)
print("Feature shape:", X_train_feat.shape)

# ====== 정규화 ======
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_val_scaled = scaler.transform(X_val_feat)

# ====== 라벨 인코딩 ======
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

# ====== LDA 적용 (지도 학습 기반 차원 축소) ======
lda = LDA(n_components=9)  # 클래스 수 - 1
X_train_lda = lda.fit_transform(X_train_scaled, y_train_enc)
X_val_lda = lda.transform(X_val_scaled)

# ====== KNN 학습 및 평가 ======
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train_lda, y_train_enc)

distances, indices = knn.kneighbors(X_val_lda, n_neighbors=K)

correct_count = 0
top10_labels_list = []

for i in range(len(indices)):
    top10_label_indices = y_train_enc[indices[i]]
    top10_label_names = list(le.inverse_transform(top10_label_indices))
    top10_labels_list.append(','.join(top10_label_names))

    is_all_same = np.all(top10_label_indices == y_val_enc[i])
    if is_all_same:
        correct_count += 1

top10_accuracy = correct_count / len(indices)
print(f"Task 2 - Validation Top-10 Retrieval Accuracy (LDA): {top10_accuracy * 100:.2f}%")

# ====== 결과 저장 ==
with open('c1_t2_a1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['QueryImage'] + [f'Top{i + 1}' for i in range(K)])
    for i, top10_labels_str in enumerate(top10_labels_list):
        top10_labels = top10_labels_str.split(',')
        row = [f'query{i + 1:03}.png'] + top10_labels
        writer.writerow(row)

print("CSV 저장 완료: c1_t2_a1.csv")
