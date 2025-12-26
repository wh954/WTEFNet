import argparse

from ultralytics import YOLO


def main(opt):
    print(" 当前使用的模型 config 是：", opt.model_path)
    with open(opt.model_path) as f:
        config_text = f.read()
        print("当前模型 config 内容如下：\n", config_text)
    model = YOLO(opt.model_path, task="detect")

    model.train(
        data=opt.data_path,
        epochs=opt.epochs,
        batch=opt.batch_size,
        imgsz=opt.img_size,
        device=opt.device,
        workers=opt.workers,
        patience=500,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="cfg/models/v10/yolov10s.yaml")
    parser.add_argument("--data_path", type=str, default="data/mydata.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=16)
    opt = parser.parse_args()

    main(opt)
