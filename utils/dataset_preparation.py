import shutil
import cv2
from pathlib import Path


class DatasetPreparator:
    """Подготовка датасета YOLO из разметки CVAT."""

    CLASS_NAMES = {0: "gun", 1: "person"}

    def __init__(self, project_root="."):
        self.root = Path(project_root).resolve()
        self.data_dir = self.root / "data"

    def create_folders(self):
        folders = [
            "images/train",
            "images/val",
            "labels/train",
            "labels/val",
        ]
        for folder in folders:
            (self.data_dir / folder).mkdir(parents=True, exist_ok=True)

    def split_video_to_frames(self, video_path, output_dir, frame_interval=10):
        """Извлекает кадры из видео."""
        cap = cv2.VideoCapture(str(video_path))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_name = f"frame_{saved_count:06d}.png"
                cv2.imwrite(str(output_dir / frame_name), frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Извлечено {saved_count} кадров из {frame_count}")

    def prepare_dataset(self):
        """
        Мигрирует данные в стандартную структуру YOLO:
        - data/images/test -> data/images/train
        - data/annotations/test -> data/labels/train
        - dataset/obj_Train_data -> data/labels/val
        """
        self.create_folders()

        train_images_src = self.data_dir / "images" / "test"
        val_images_dir = self.data_dir / "images" / "val"
        train_images_dst = self.data_dir / "images" / "train"
        train_labels_src = self.data_dir / "annotations" / "test"
        train_labels_dst = self.data_dir / "labels" / "train"
        val_labels_src = self.root / "dataset" / "obj_Train_data"
        val_labels_dst = self.data_dir / "labels" / "val"

        if train_images_src.exists():
            moved = 0
            for img in train_images_src.glob("*.png"):
                dst = train_images_dst / img.name
                if not dst.exists():
                    shutil.move(str(img), str(dst))
                moved += 1
            print(f"Перемещено изображений train: {moved}")

        if train_labels_src.exists():
            copied = 0
            for label in train_labels_src.glob("*.txt"):
                img_stem = label.stem
                img_path = train_images_dst / f"{img_stem}.png"
                if img_path.exists():
                    shutil.copy2(label, train_labels_dst / label.name)
                    copied += 1
            print(f"Скопировано меток train: {copied}")

        if val_labels_src.exists() and val_images_dir.exists():
            copied = 0
            for img in val_images_dir.glob("*.png"):
                label_src = val_labels_src / f"{img.stem}.txt"
                if label_src.exists():
                    shutil.copy2(label_src, val_labels_dst / label_src.name)
                    copied += 1
            print(f"Скопировано меток val: {copied}")

        self._validate_pairs()
        self.create_yaml_config()
        self._cleanup_legacy()

    def _validate_pairs(self):
        for split in ("train", "val"):
            images_dir = self.data_dir / "images" / split
            labels_dir = self.data_dir / "labels" / split
            images = list(images_dir.glob("*.png"))
            missing = sum(
                1 for img in images if not (labels_dir / f"{img.stem}.txt").exists()
            )
            print(f"{split}: {len(images)} изображений, без меток: {missing}")

    def _cleanup_legacy(self):
        legacy_dirs = [
            self.data_dir / "images" / "test",
            self.data_dir / "annotations",
            self.root / "dataset",
        ]
        for path in legacy_dirs:
            if path.exists():
                shutil.rmtree(path)
                print(f"Удалено: {path}")

        duplicate_video = self.data_dir / "videos"
        if duplicate_video.exists():
            shutil.rmtree(duplicate_video)
            print(f"Удалено: {duplicate_video}")

        old_runs = self.root / "runs" / "train"
        if old_runs.exists():
            for exp in old_runs.iterdir():
                if exp.is_dir() and not (exp / "weights").exists():
                    shutil.rmtree(exp)
                    print(f"Удалён пустой эксперимент: {exp}")

    def create_yaml_config(self):
        config_path = self.data_dir / "data.yaml"
        content = f"""path: .
train: data/images/train
val: data/images/val

nc: 2
names:
  0: gun
  1: person
"""
        config_path.write_text(content, encoding="utf-8")
        print(f"Создан {config_path}")


if __name__ == "__main__":
    preparator = DatasetPreparator(".")
    preparator.prepare_dataset()
