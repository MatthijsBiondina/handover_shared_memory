from ultralytics import YOLO
from cantrips.logging.logger import get_logger
from machine_learning.utils import make_ego_dataset, make_yolo_dataset

logger = get_logger()


def main():
    # make_yolo_dataset(
    #     in_dir="/home/matt/Datasets/COCO-Hand/COCO-Hand-Big",
    #     ou_dir="/home/matt/Datasets/YOLO-Hand",
    # )
    make_ego_dataset(
        in_dir="/home/matt/Datasets/egohands_data/samples",
        ou_dir="/home/matt/Datasets/YOLO-Ego"
    )

    # model = YOLO("yolov8n.pt")
    # model.train(
    #     data="/home/matt/Datasets/YOLO-Hand/manifest.yaml",
    #     epochs=10,
    #     imgsz=640,
    #     batch=64,
    # )


if __name__ == "__main__":
    main()
