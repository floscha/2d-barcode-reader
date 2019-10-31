![barcode reader logo](logo.png)

# 2D Barcode Reader

[![Build Status](https://travis-ci.org/floscha/2d-barcode-reader.svg?branch=master)](https://travis-ci.org/floscha/2d-barcode-reader)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d85fb6267312422baa08c6c8385da846)](https://www.codacy.com/app/floscha/2d-barcode-reader?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=floscha/2d-barcode-reader&amp;utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python/OpenCV-based barcode reader for 2D barcode markers.


## Example

Assume we have an image containing one or several markers like the one below:

![clean marker](example/clean_marker.png)

To extract the marker(s) we can use the following script similar to the [example reader](example/example_reader.py):


```python
import cv2

from barcode_reader import MarkerDetector

image_path = 'clean_marker.png'
frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

if frame is None:
    raise ValueError("Image could not be read")

detector = MarkerDetector(min_contour_length_allowed=10000)
detected_markers = detector.process_frame(frame)

print("%d markers detected:" % len(detected_markers))
for marker in detected_markers:
    print(marker.points)
```

This outputs the detected contour(s).

```
2 markers detected:
[[119.43042755 120.34486389]
 [380.50708008 120.37145233]
 [380.46075439 379.82989502]
 [119.41880798 379.66653442]]
[[119.43190002 120.3412323 ]
 [380.51220703 120.37619019]
 [380.45599365 379.8331604 ]
 [119.41361237 379.66433716]]
```

Also, when debugging is enabled, it draws-in the contours like shown below:
![marker with contours](example/debug.png)
