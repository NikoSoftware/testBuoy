
from ultralytics import YOLO


def run():
    model = YOLO("./yolo11m.pt")


    train_results = model.train(
        data="buoy.yaml",
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image size for training
        device="cuda",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        batch=10,
        save=True,
        save_period=40,
        project="runs/train",
        workers=20
    )


if __name__ == '__main__':

    run()
