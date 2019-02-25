import sys

import cv2

from barcode_reader import MarkerDetector


if __name__ == '__main__':
    args = sys.argv[1:]

    image_path = args[0]
    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Image could not be read")

    detector = MarkerDetector(min_contour_length_allowed=10000)
    detected_markers = detector.process_frame(frame, markers_only=True)

    print("%d markers detected:" % len(detected_markers))
    for marker in detected_markers:
        current_marker_id = marker.get
        print(current_marker_id)
