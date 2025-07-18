from ultralytics import YOLO
import cv2
import os

# Load the YOLO model (change this to your own trained model path)
MODEL_PATH = "yolov8m.pt"  # Replace with your custom mango detector

# Initialize model
model = YOLO(MODEL_PATH)


def detect_and_crop_mangoes(image_path, conf_threshold=0.4):
    """
    Detects mangoes in the input image using YOLO and returns cropped mango images.
    :param image_path: Path to the input image
    :param conf_threshold: Confidence threshold for detection
    :return: List of cropped mango image arrays
    """
    results = model(image_path)[0]  # First image only

    img = cv2.imread(image_path)
    crops = []

    for box in results.boxes:
        conf = float(box.conf)
        if conf < conf_threshold:
            continue

        # Get coordinates and convert to int
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        if crop.size > 0:
            crops.append(crop)

    return crops


# Optional: test run
if __name__ == "__main__":
    test_image = "mango\\dataset.v18i.tensorflow\\test\\1-2-_jpg.rf.1345445b10ba17bdbd26ea13448d1f7b.jpg"
    crops = detect_and_crop_mangoes(test_image)

    for i, crop in enumerate(crops):
        cv2.imwrite(f"mango_blocks/detected_{i}.jpg", crop)
        print(f"Saved mango crop {i}")
