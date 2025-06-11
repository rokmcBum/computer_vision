import csv
import os

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

# ====== 설정 ======
TRAIN_DIR = "Large"
TEST_DIR = "test"
IMAGE_SIZE = (256, 256)
HOG_PIXELS_PER_CELL = (32, 32)
K = 10
print("IMAGE_SIZE:", IMAGE_SIZE, "HOG:", HOG_PIXELS_PER_CELL)

CLASS_NAMES = [
    'Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney',
    'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light'
]


# ====== 이미지 로딩 ======
def load_images(folder_path, valid_classes=None, size=(64, 64), use_subdirs=True):
    images, labels, filenames = [], [], []
    if use_subdirs:
        for class_name in os.listdir(folder_path):
            if valid_classes and class_name not in valid_classes:
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
                filenames.append(filename)
    else:
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, size)
            images.append(img)
            filenames.append(filename)
    return np.array(images), (np.array(labels) if use_subdirs else None), filenames


# ====== Feature 추출 함수 (HOG + LBP + Histogram) ======
def extract_combined_features(images, pixels_per_cell=(8, 8)):
    features = []
    for img in tqdm(images, desc="Extracting Features"):
        hog_feature = hog(
            img, orientations=9,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
        (hist_lbp, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, 11),
                                     range=(0, 10))
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-7)
        hist_gray = cv2.calcHist([img], [0], None, [16], [0, 256])
        hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()
        combined = np.concatenate([hog_feature, hist_lbp, hist_gray])
        features.append(combined)
    return np.array(features)


# ====== 학습 데이터 처리 ======
X_train_img, y_train, _ = load_images(TRAIN_DIR, valid_classes=CLASS_NAMES, size=IMAGE_SIZE, use_subdirs=True)
print("Train 이미지 개수:", len(X_train_img))
X_train_feat = extract_combined_features(X_train_img, pixels_per_cell=HOG_PIXELS_PER_CELL)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

lda = LDA(n_components=9)
X_train_lda = lda.fit_transform(X_train_scaled, y_train_enc)

# ====== 테스트 데이터 처리 ======
X_test_img, _, test_filenames = load_images(TEST_DIR, size=IMAGE_SIZE, use_subdirs=False)
print("Test 이미지 개수:", len(X_test_img))
X_test_feat = extract_combined_features(X_test_img, pixels_per_cell=HOG_PIXELS_PER_CELL)
X_test_scaled = scaler.transform(X_test_feat)
X_test_lda = lda.transform(X_test_scaled)

# ====== KNN 분류 및 Top-K 예측 ======
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train_lda, y_train_enc)
indices = knn.kneighbors(X_test_lda, n_neighbors=K, return_distance=False)

# ====== 결과 저장 ======
with open('c1_t2_a1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['QueryImage'] + [f'Top{i + 1}' for i in range(K)])
    for fname, neighbor_indices in zip(test_filenames, indices):
        topK_labels = le.inverse_transform(y_train_enc[neighbor_indices])
        writer.writerow([fname] + list(topK_labels))

print("✅ CSV 저장 완료: c1_t2_a1.csv")
