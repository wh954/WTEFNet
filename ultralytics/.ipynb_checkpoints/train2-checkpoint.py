import argparse
from ultralytics import YOLO

def main(opt):
    model = YOLO(opt.model_path)

    results = model.train(
        data=opt.data_path,
        epochs=opt.epochs,
        batch=opt.batch_size,
        imgsz=opt.img_size,
        device=opt.device,
        workers=opt.workers,
        patience=500  # ← 添加这里！
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/root/ultralytics-main/ultralytics/cfg/models/v8/yolov8.yaml')
    parser.add_argument('--data_path', type=str, default='/root/ultralytics-main/ultralytics/data/mydata2.yaml')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=16)
    opt = parser.parse_args()

    main(opt)
