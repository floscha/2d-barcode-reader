import sys
sys.path.append('../src')

import cv2

from marker_detector import MarkerDetector


if __name__ == '__main__':
    args = sys.argv[1:]

    img_fpath = args[0]
    frame = cv2.imread(img_fpath, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Image could not be read")

    detector = MarkerDetector()
    detected_markers = detector.process_frame(frame, markers_only=True)

    print("%d markers detected:" % len(detected_markers))
    for marker in detected_markers:
        current_marker_id = marker.id
        print(current_marker_id)
