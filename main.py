import argparse
from pathlib import Path

from detect_video import GunPersonDetector
from train import ModelTrainer
from utils.dataset_preparation import DatasetPreparator


def main():
    parser = argparse.ArgumentParser(description="Person and Gun Detection System")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["prepare", "train", "detect", "realtime"],
        help="Режим работы: prepare, train, detect, realtime",
    )
    parser.add_argument("--video", type=str, help="Путь к видео файлу")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Путь к модели",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Количество эпох для обучения",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Размер батча (по умолчанию подбирается под VRAM GPU)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Показывать видео при детекции",
    )

    args = parser.parse_args()

    if args.mode == "prepare":
        print("Подготовка датасета...")
        preparator = DatasetPreparator(".")
        preparator.prepare_dataset()
        print("Датасет подготовлен!")

    elif args.mode == "train":
        print("Начало обучения...")
        trainer = ModelTrainer("data/data.yaml")
        trainer.train_model(epochs=args.epochs, batch=args.batch)
        trainer.evaluate_model("models/best.pt")
        print("Обучение завершено!")

    elif args.mode == "detect":
        if not args.video:
            print("Ошибка: укажите путь к видео файлу --video")
            return

        if not Path(args.video).exists():
            print(f"Видео не найдено: {args.video}")
            return

        if not Path(args.model).exists():
            print(f"Модель не найдена: {args.model}")
            return

        print("Запуск детекции с трекингом...")
        detector = GunPersonDetector(args.model)
        video_name = Path(args.video).stem
        output_path = f"outputs/{video_name}_detected.mp4"
        detector.process_video(args.video, output_path, show=args.show)

    elif args.mode == "realtime":
        if not Path(args.model).exists():
            print(f"Модель не найдена: {args.model}")
            return

        print("Запуск детекции в реальном времени...")
        detector = GunPersonDetector(args.model)
        detector.process_realtime()


if __name__ == "__main__":
    main()
