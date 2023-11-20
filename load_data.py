from datasets import load_dataset, load_from_disk
from vit import config
import cv2
import matplotlib.pyplot as plt

def load_data_beans():
    ds = load_dataset(
        path="beans",
        split="train"
    )
    ds.save_to_disk(config.IMAGES_TRAIN_PATH)

    ds = load_dataset(
        path="beans",
        split="validation"
    )
    ds.save_to_disk(config.IMAGES_VAL_PATH)

    ds = load_dataset(
        path="beans",
        split="test"
    )
    ds.save_to_disk(config.IMAGES_TEST_PATH)


ds = load_from_disk("./dataset/train")
print(ds[0])

print(ds[0]["image"])

image = cv2.imread(ds[0]["image_file_path"])
print(image)
image = image.astype("float32") / 255.0
plt.imshow(image)
plt.show()