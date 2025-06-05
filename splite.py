import os
import shutil

from sklearn.model_selection import train_test_split

# ====== 설정 ======
SOURCE_DIR = "Splited"
TARGET_DIR = "a"
TRAIN_RATIO = 0.9

# ====== 클래스 목록 가져오기 ======
class_names = sorted(os.listdir(SOURCE_DIR))
print("클래스:", class_names)

# ====== train/test 디렉토리 생성 ======
for split in ['train', 'test']:
    for cls in class_names:
        os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

# ====== 클래스별 이미지 나누기 ======
for cls in class_names:
    cls_path = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    train_imgs, test_imgs = train_test_split(images, train_size=TRAIN_RATIO, random_state=42)

    # train 저장
    for img in train_imgs:
        src = os.path.join(cls_path, img)
        dst = os.path.join(TARGET_DIR, 'train', cls, img)
        shutil.copy2(src, dst)

    # test 저장
    for img in test_imgs:
        src = os.path.join(cls_path, img)
        dst = os.path.join(TARGET_DIR, 'test', cls, img)
        shutil.copy2(src, dst)

print("✔️ 데이터 분할 및 저장 완료!")
print(f"→ {TARGET_DIR}/train/[class]/")
print(f"→ {TARGET_DIR}/test/[class]/")
