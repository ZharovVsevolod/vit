from datasets import load_dataset, load_from_disk
from vit import config
import cv2
import matplotlib.pyplot as plt
import torch
import einops

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

# load_data_beans()

ds = load_from_disk("./dataset/train")
# print(ds[0])
# print(ds[0]["image"])

image = cv2.imread(ds[0]["image_file_path"])
image = image.astype("float64") / 255.0
plt.imshow(image)
plt.show()

image_t = torch.tensor(image)
image_t = einops.rearrange(image_t, "h w c -> c w h", c=3)
print(image_t.shape)