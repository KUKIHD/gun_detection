import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


class ModelTrainer:
    EXPERIMENT_NAME = "gun_person_v1"

    def __init__(self, config_path="data/data.yaml", require_gpu=True):
        self.config_path = config_path
        self.require_gpu = require_gpu

        if torch.cuda.is_available():
            self.device = 0
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            print(f"Используется устройство: cuda:0")
        else:
            if require_gpu:
                raise RuntimeError(
                    "CUDA недоступна. Установите PyTorch с поддержкой GPU:\n"
                    "  pip uninstall torch torchvision -y\n"
                    "  pip install torch torchvision --index-url "
                    "https://download.pytorch.org/whl/cu124"
                )
            self.device = "cpu"
            print("ВНИМАНИЕ: обучение на CPU (медленно)")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    @staticmethod
    def default_batch_size():
        if not torch.cuda.is_available():
            return 8
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 10:
            return 32
        if vram_gb >= 8:
            return 24
        if vram_gb >= 6:
            return 24  # GTX 1660 6GB: ~5 GB VRAM, быстрее чем batch=8
        if vram_gb >= 4:
            return 16
        return 8

    def train_model(
        self,
        model_name="yolov8n.pt",
        epochs=100,
        imgsz=640,
        batch=None,
        patience=30,
    ):
        if batch is None:
            batch = self.default_batch_size()
        print(f"Начинаем обучение модели {model_name}...")
        print(f"Классы: {self.config['names']}")
        print(f"Batch size: {batch}")

        Path("models").mkdir(exist_ok=True)

        model = YOLO(model_name)
        results = model.train(
            data=self.config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            device=self.device,
            workers=4,
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            save=True,
            save_period=10,
            pretrained=True,
            verbose=True,
            project="runs/train",
            name=self.EXPERIMENT_NAME,
            exist_ok=True,
        )

        print("Обучение завершено!")

        save_dir = Path(results.save_dir)
        best_model_path = save_dir / "weights" / "best.pt"
        if best_model_path.exists():
            shutil.copy(best_model_path, "models/best.pt")
            print("Лучшая модель сохранена в models/best.pt")
        else:
            print(f"Внимание: не найден {best_model_path}")

        return results

    def evaluate_model(self, model_path="models/best.pt"):
        model = YOLO(model_path)
        metrics = model.val(data=self.config_path)

        print("\n=== Результаты оценки ===")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        p = metrics.box.p
        r = metrics.box.r
        print(f"Precision: {float(p.mean() if hasattr(p, 'mean') else p):.4f}")
        print(f"Recall: {float(r.mean() if hasattr(r, 'mean') else r):.4f}")

        return metrics


if __name__ == "__main__":
    trainer = ModelTrainer("data/data.yaml")
    trainer.train_model(
        model_name="yolov8n.pt",
        epochs=100,
        imgsz=640,
        patience=30,
    )
    trainer.evaluate_model("models/best.pt")
