import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


class GunPersonDetector:
    def __init__(self, model_path="models/best.pt"):
        print(f"Загрузка модели: {model_path}")
        self.model = YOLO(model_path)
        self.colors = {
            "gun": (0, 0, 255),
            "person": (0, 255, 0),
        }
        self.confidence_threshold = 0.5
        print("Модель загружена успешно")

    def process_frame(self, frame):
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []
        processed_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[cls_id]

                detections.append(
                    {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

                color = self.colors.get(class_name, (255, 255, 255))
                cv2.rectangle(
                    processed_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2,
                )

                label = f"{class_name}: {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    processed_frame,
                    (int(x1), int(y1) - text_height - 10),
                    (int(x1) + text_width, int(y1)),
                    color,
                    -1,
                )
                cv2.putText(
                    processed_frame,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        return processed_frame, detections

    def process_video(self, video_path, output_path=None, show=False):
        print(f"Обработка видео: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Не удалось открыть видео: {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Параметры видео: {width}x{height}, {fps} FPS")

        out = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()
        gun_detections = 0
        person_detections = 0

        print("Начинаю обработку видео...")
        if show:
            print("Нажмите 'q' для выхода из превью")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, detections = self.process_frame(frame)

            for detection in detections:
                if detection["class"] == "gun":
                    gun_detections += 1
                elif detection["class"] == "person":
                    person_detections += 1

            stats_text = (
                f"Guns: {gun_detections} | Persons: {person_detections} | "
                f"Frame: {frame_count}"
            )
            cv2.putText(
                processed_frame,
                stats_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if out is not None:
                out.write(processed_frame)

            if show:
                cv2.imshow("Gun & Person Detection", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Остановлено пользователем")
                    break

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Обработано кадров: {frame_count}, "
                    f"Время: {elapsed:.1f}с, FPS: {frame_count / elapsed:.1f}"
                )

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print("\n" + "=" * 50)
        print("ИТОГОВАЯ СТАТИСТИКА:")
        print(f"   Всего кадров: {frame_count}")
        print(f"   Общее время: {total_time:.2f} секунд")
        print(f"   Средний FPS: {avg_fps:.1f}")
        print(f"   Обнаружено оружия: {gun_detections}")
        print(f"   Обнаружено людей: {person_detections}")
        print("=" * 50)

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        if output_path:
            print(f"Результат сохранен: {output_path}")

    def process_realtime(self, camera_id=0):
        print(f"Запуск детекции с камеры {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Не удалось открыть камеру: {camera_id}")
            return

        print("Нажмите 'q' для выхода")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, _ = self.process_frame(frame)
            cv2.imshow("Gun & Person Detection (Realtime)", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Детекция людей и оружия на видео")
    parser.add_argument("--video", type=str, required=True, help="Путь к входному видео")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь для сохранения результата",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Путь к модели",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Показывать видео в реальном времени",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Порог уверенности",
    )

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Видео файл не найден: {args.video}")
        return

    if not Path(args.model).exists():
        print(f"Файл модели не найден: {args.model}")
        print("Сначала обучите модель: python main.py --mode train")
        return

    output_path = args.output
    if output_path is None:
        video_name = Path(args.video).stem
        output_path = f"outputs/{video_name}_detected.mp4"

    detector = GunPersonDetector(args.model)
    detector.confidence_threshold = args.conf
    detector.process_video(
        video_path=args.video,
        output_path=output_path,
        show=args.show,
    )


if __name__ == "__main__":
    main()
