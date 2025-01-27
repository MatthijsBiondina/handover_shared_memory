import csv
import os
from random import shuffle
import shutil
from PIL import Image
import scipy

from cantrips.debugging.terminal import pbar
from cantrips.file_management import listdir, makedirs
from cantrips.logging.logger import get_logger

logger = get_logger()


def make_yolo_dataset(in_dir, ou_dir):
    shutil.rmtree(ou_dir, ignore_errors=True)
    for col in ["train", "val", "test"]:
        makedirs(f"{ou_dir}/images/{col}")
        makedirs(f"{ou_dir}/labels/{col}")

    # Collect all images
    with open(f"{in_dir}/annotations.txt", "r") as f:
        images = list(set([row[0] for row in csv.reader(f)]))
    shuffle(images)
    images = {name: {"id": ii} for ii, name in enumerate(list(images))}

    # Assign collection
    for ii, name in enumerate(images.keys()):
        if ii < len(images.keys()) * 0.8:
            images[name]["col"] = "train"
        elif ii < len(images.keys()) * 0.9:
            images[name]["col"] = "val"
        else:
            images[name]["col"] = "test"

    # Make annotations
    with open(f"{in_dir}/annotations.txt", "r") as f:
        for row in pbar(csv.reader(f), desc="Writing Annotations"):
            name = row[0]
            xmin, xmax, ymin, ymax = map(int, row[1:5])
            img_width, img_height = Image.open(f"{in_dir}/images/{name}").size

            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            col = images[name]["col"]
            with open(
                f"{ou_dir}/labels/{col}/{name.replace('.jpg','.txt')}", "a+"
            ) as f:
                f.write(f"0 {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}\n")

    # Copy images
    for name in pbar(images.keys(), desc="Moving Images"):
        col = images[name]["col"]
        os.system(f"cp {in_dir}/images/{name} {ou_dir}/images/{col}/{name}")

    # Write manifest:
    manifest = f"""#Dataset path configuration
path: {ou_dir}
train: images/train
val: images/val

names:
  0: hand

nc: 1
"""
    with open(f"{ou_dir}/manifest.yaml", "w+") as f:
        f.write(manifest)

def make_ego_dataset(in_dir, ou_dir):
    shutil.rmtree(ou_dir, ignore_errors=True)
    for col in ["train", "val", "test"]:
        makedirs(f"{ou_dir}/images/{col}")
        makedirs(f"{ou_dir}/labels/{col}")

    for folder in listdir(in_dir):
        mat_data = scipy.io.loadmat(folder / "polygons.mat")
        logger.info(folder)
    logger.info("foo")