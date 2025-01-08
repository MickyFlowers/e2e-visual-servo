import torch
import cv2
import numpy as np

dataset = torch.load("/home/cyx/project/data/policy/sample/data/round000000.pt")
print(f"Keys: {dataset.keys()}")
data = dict()
image_like_keys = [
    "desired_rgbs",
    "current_rgbs",
    "desired_pcds",
    "current_pcds",
]
for k, v in dataset.items():
    data[k] = v[0]
    if k in image_like_keys:
        data[k] = data[k].permute(1, 2, 0).cpu().numpy()
    else:
        data[k] = data[k].cpu().numpy()
print(f"obj_num_changed: {data['obj_num_changed']}")
for i, k in enumerate(image_like_keys):
    if i == 0:
        concat_image = data[image_like_keys[i]]
    else:
        concat_image = np.concatenate((concat_image, data[image_like_keys[i]]), axis=1)

desired_masked_image = data["desired_rgbs"] * np.expand_dims(
    data["desired_masks"], axis=-1
)
current_masked_image = data["current_rgbs"] * np.expand_dims(
    data["current_masks"], axis=-1
)
masked_image = np.concatenate((desired_masked_image, current_masked_image), axis=1)
cv2.imshow("masked_images", masked_image)
cv2.imshow("concat_image", concat_image)
cv2.waitKey(0)
