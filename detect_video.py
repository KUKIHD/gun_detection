import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class GunPersonDetector:
    def __init__(self, model_path="models/best.pt", tracker="bytetrack.yaml"):
        print(f"Загрузка модели: {model_path}")
        self.model = YOLO(model_path)
        self.tracker = tracker
        self.colors = {
            "gun": (0, 0, 255),
            "person": (0, 255, 0),
        }
        self.confidence_threshold = 0.5
        self._track_colors = {}
        print("Модель загружена успешно (трекинг: ByteTrack)")

    def _track_color(self, track_id):
        if track_id not in self._track_colors:
            hue = (track_id * 47) % 180
            hsv = np.uint8([[[hue, 200, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            self._track_colors[track_id] = tuple(int(c) for c in bgr)
        return self._track_colors[track_id]

    def process_frame(self, frame, track=True):
        if track:
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                persist=True,
                tracker=self.tracker,
                verbose=False,
            )
        else:
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
                track_id = int(box.id[0]) if box.id is not None else None

                detections.append(
                    {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "track_id": track_id,
                    }
                )

                if track_id is not None:
                    color = self._track_color(track_id)
                else:
                    color = self.colors.get(class_name, (255, 255, 255))

                cv2.rectangle(
                    processed_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2,
                )

                if track_id is not None:
                    label = f"{class_name} #{track_id}: {confidence:.2f}"
                else:
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

    def process_video(self, video_path, output_path=None, show=False, track=True):
        print(f"Обработка видео: {video_path}")
        if track:
            print("Режим: детекция + трекинг (ByteTrack)")

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
            if not out.isOpened():
                print(f"Не удалось создать выходной файл: {output_path}")
                cap.release()
                return

        frame_count = 0
        start_time = time.time()
        gun_track_ids = set()
        person_track_ids = set()

        print("Начинаю обработку видео...")
        if show:
            print("Нажмите 'q' для выхода из превью")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, detections = self.process_frame(frame, track=track)

            active_guns = 0
            active_persons = 0
            for detection in detections:
                track_id = detection.get("track_id")
                if detection["class"] == "gun":
                    active_guns += 1
                    if track_id is not None:
                        gun_track_ids.add(track_id)
                elif detection["class"] == "person":
                    active_persons += 1
                    if track_id is not None:
                        person_track_ids.add(track_id)

            stats_text = (
                f"Guns: {active_guns} ({len(gun_track_ids)} tracks) | "
                f"Persons: {active_persons} ({len(person_track_ids)} tracks) | "
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
                cv2.imshow("Gun & Person Detection + Tracking", processed_frame)
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
        print(f"   Уникальных треков gun: {len(gun_track_ids)}")
        print(f"   Уникальных треков person: {len(person_track_ids)}")
        print("=" * 50)

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        if output_path:
            print(f"Результат сохранен: {output_path}")

    def process_realtime(self, camera_id=0):
        print(f"Запуск детекции с трекингом с камеры {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Не удалось открыть камеру: {camera_id}")
            return

        print("Нажмите 'q' для выхода")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, _ = self.process_frame(frame, track=True)
            cv2.imshow("Gun & Person Detection + Tracking (Realtime)", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Детекция и трекинг людей и оружия на видео")
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
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Отключить трекинг (только детекция по кадрам)",
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
        track=not args.no_track,
    )


if __name__ == "__main__":
    main()
