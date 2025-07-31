from ultralytics import YOLO
import pandas as pd

model = YOLO("/home/signal/PycharmProjects/bdd100k/cover_ultralytics-main/ultralytics/runs/detect/train/weights/best.pt")


metrics = model.val(
    data="/home/signal/PycharmProjects/bdd100k/cover_ultralytics-main/ultralytics/data/mydata.yaml",
    imgsz=640,
    device=0,
    split="test",
    save_json=True
)

df = pd.DataFrame(metrics.results_dict, index=[0])

out_csv = "val_results_test.csv"
df.to_csv(out_csv, index=False)
print(f"CSV 已写入 {out_csv}")