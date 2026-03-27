
from pathlib import Path
import  matplotlib.pyplot as plt
import  random

data_dir=Path('data/mvtec_loco_anomaly_detection')
path=Path(__file__).resolve().parent.parent/data_dir
cls_path= path / "juice_bottle"
print(list(cls_path.iterdir()))
print(list((cls_path / "train").iterdir()))
print(len(list((cls_path / "train" / "good").glob("*.png")))
)

img_paths = list((cls_path / "train" / "good").glob("*.png"))

sample_imgs = random.sample(img_paths, 5)


plt.figure(figsize=(10,4))
for i,img_path in enumerate(sample_imgs):
    img=plt.imread(img_path)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')
plt.show()