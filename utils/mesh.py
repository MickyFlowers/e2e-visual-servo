import trimesh
import numpy as np
import trimesh.visual
from PIL import Image


def create_image_mesh(size=(1.0, 1.0), image_path: str = None):

    h, w = size
    vertices = np.array(
        [
            [size[0], 0.0, 0.0],
            [size[0], size[1], 0.0],
            [0.0, 0.0, 0.0],
            [0.0, size[1], 0.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
        ],
    )
    uv = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.uv = uv
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")

        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
    return mesh
