"""This file defines the MarkerDetector class."""
import cv2
import numpy as np

from marker import Marker


class MarkerDetector(object):
    """A detector class to retrieve markers from an image."""

    def __init__(self, calibration=None):
        """Initialize a new MarkerDetector object."""
        self.min_contour_length_allowed = 100
        self.marker_size = (100, 100)

        if calibration:
            self.cam_matrix = calibration.m_intrinsic
            self.dist_coeff = calibration.m_distorsion
        else:
            print("Warning: No calibration given. " +
                  "Pose estimation unavailable.")
            self.cam_matrix = None
            self.dist_coeff = None

        # Define marker corners.
        self.m_mc_3d = np.array([[-0.5, -0.5, 0],
                                 [0.5,  -0.5, 0],
                                 [0.5,   0.5, 0],
                                 [-0.5,  0.5, 0]],
                                dtype=np.float32)
        self.m_mc_3d *= 5  # Why?!

        self.m_mc_2d = np.array(
            [[0,                       0],
             [self.marker_size[0] - 1, 0],
             [self.marker_size[0] - 1, self.marker_size[1] - 1],
             [0,                       self.marker_size[1] - 1]],
            dtype=np.float32
        )

    def process_frame(self, frame, markers_only=False):
        """Estimate the 3D pose for a frame using marker detection.

        It thereby uses the following 6 steps:
            1. Convert the input image to grayscale
            2. Perform binary threshold operation
            3. Detect contours
            4. Search for possible markers
            5. Detect and decode markers
            6. Estimate marker 3D pose
        """
        debug = 0

        # 1.Convert the input image to grayscale
        grayscale = self._prepare_image(frame)
        if debug == 1:
            frame[:, :] = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

        # 2.Perform binary threshold operation
        threshold_img = self._perform_threshold(grayscale)
        if debug == 2:
            frame[:, :] = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)

        # 3.Detect contours
        # contours = self.find_contours(threshold_img, grayscale.shape[0]/5)
        contours = self._find_contours(threshold_img, 5)
        if debug == 3:
            cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

        # 4.Search for possible markers
        possible_markers = self._find_marker_candidates(contours)
        if debug == 4:
            color = (0, 0, 255)
            thickness = 2
            for m in possible_markers:
                points = [tuple(p) for p in m.points]
                cv2.line(frame, points[0], points[1], color, thickness)
                cv2.line(frame, points[1], points[2], color, thickness)
                cv2.line(frame, points[2], points[3], color, thickness)
                cv2.line(frame, points[3], points[0], color, thickness)

        # 5.Detect and decode markers
        detected_markers = self._detect_markers(grayscale, possible_markers)

        if markers_only:
            return detected_markers

        # 6.Estimate marker 3D pose
        if not self.cam_matrix or not self.dist_coeff:
            raise ValueError("No camera calibration data available.")

        return self._estimate_position(detected_markers)

    def _prepare_image(self, img):
        """Convert the given image to grayscale."""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _perform_threshold(self, img):
        """Perform binary threshold operation on the given image."""
        img = cv2.GaussianBlur(img,
                               ksize=(5, 5),
                               sigmaX=0)
        return cv2.adaptiveThreshold(
            img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=7,
            C=7
        )

    def _find_contours(self, img, min_contour_points_allowed):
        """Find all contours within the given image."""
        contours, hierarchy = cv2.findContours(img,
                                               mode=cv2.RETR_LIST,
                                               method=cv2.CHAIN_APPROX_NONE)
        return [c for c in contours if len(c) >= min_contour_points_allowed]

    def _find_marker_candidates(self, contours):
        """Search for possible markers amongst a number of contours."""
        # For each contour: decide if it's a possible marker.
        possible_markers = []
        for c in contours:
            # Approximate to polygon.
            eps = len(c) * 0.05
            approx_curve = cv2.approxPolyDP(c, eps, True)
            # Only consider polygons that have 4 points and are convex.
            if not len(approx_curve) == 4:
                continue
            if not cv2.isContourConvex(approx_curve):
                continue

            # Distance between points has to be large enough.
            min_dist = float("inf")
            for i in range(4):
                side = approx_curve[i] - approx_curve[(i + 1) % 4]
                side = side[0]
                squared_side_length = np.dot(side, side)
                min_dist = min(min_dist, squared_side_length)
            if min_dist < self.min_contour_length_allowed:
                continue

            # Prepare marker for saving.
            m = [np.array([p[0][0], p[0][1]]) for p in approx_curve]
            v1 = m[1] - m[0]
            v2 = m[2] - m[0]
            o = (v1[0] * v2[1]) - (v1[1] * v2[0])
            if o < 0:
                m[1], m[3] = m[3], m[1]
            possible_markers.append(np.array(m))

        # Mask elements with corners too close to each other.
        too_near_candidates = []
        for i in range(len(possible_markers)):
            m1 = possible_markers[i]
            for j in range(i+1, len(possible_markers)):
                m2 = possible_markers[j]
                dist_squared = 0
                for c in range(4):
                    v = m1[c] - m2[c]
                    dist_squared += np.dot(v, v)
                dist_squared /= 4

                if dist_squared < 100:
                    too_near_candidates.append((i, j))

        # Mask element with smaller perimeter.
        removal_mask = np.zeros(len(possible_markers))
        for i in range(len(too_near_candidates)):
            p1 = cv2.arcLength(possible_markers[too_near_candidates[i][0]],
                               True)
            p2 = cv2.arcLength(possible_markers[too_near_candidates[i][1]],
                               True)
            if p1 > p2:
                removal_index = too_near_candidates[i][0]
            else:
                removal_index = too_near_candidates[i][1]
            removal_mask[removal_index] = 1

        # Return only unmasked elements.
        detected_markers = []
        for i in range(len(possible_markers)):
            if not removal_mask[i]:
                detected_markers.append(Marker(possible_markers[i]))

        return detected_markers

    def _detect_markers(self, grayscale, marker_candidates):
        """Detect and decode markers from a list of candidates."""
        good_markers = []

        for marker in marker_candidates:
            # Remove perspective projection.
            h = self.m_mc_2d
            M = cv2.getPerspectiveTransform(marker.points, h)
            marker_image = cv2.warpPerspective(grayscale, M, self.marker_size)

            n_rotations = 0
            id_ = marker.get_marker_id(marker_image, n_rotations)
            if True:  # id:
                # marker.id = id
                # Sort the points so that they are always in the same order
                # no matter the camera orientation.
                rot = (4 - n_rotations) % 4
                marker.points = np.array(marker.points[rot:].tolist() +
                                         marker.points[:-rot].tolist())
                good_markers.append(marker)

        # Refine marker corners using sub pixel accuracy.
        if good_markers:
            precise_corners = []
            for i in range(len(good_markers)):
                marker = good_markers[i]
                for c in range(4):
                    precise_corners.append(marker.points[c])
            precise_corners = np.array(precise_corners, np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.1)
            cv2.cornerSubPix(grayscale,
                             precise_corners,
                             (5, 5),
                             (-1, -1),
                             criteria)

            # Copy back.
            for i in range(len(good_markers)):
                marker = good_markers[i]
                for c in range(4):
                    marker.points[c] = precise_corners[i * 4 + c]

        return good_markers

    def _estimate_position(self, detected_markers):
        """Estimate a 3D pose based on the detected markers."""
        for m in detected_markers:
            # x0, y0, x1, y1 = m.points[3][0], m.points[3][1], m.points[1][0],
            # m.points[1][1]
            # uad_3d = np.float32([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0],
            # [x0, y1, 0]])
            ret, rvec, tvec = cv2.solvePnP(self.m_mc_3d,
                                           m.points,
                                           self.cam_matrix,
                                           self.dist_coeff)

            R = cv2.Rodrigues(rvec)[0]
            # R = np.linalg.inv(R) #R.T rotation of inverse
            # tvec = -R * tvec # translation of inverse

            T = np.empty((4, ascii4), np.float32)
            # copy to transformation matrix
            for row in range(3):
                for col in range(3):
                    # copy rotation component
                    T[row][col] = R[row][col]
                # copy translation component
                T[col][3] = tvec[row]

            # Make matrix T homogeneous.
            # FIXME Use different x indices?! Also use NumPy op instead.
            T[0][3] = T[0][3] = T[0][3] = 0
            T[3][3] = 1

            # Convert from OpenCV to OpenGL dimensionality.
            cv_to_gl = np.zeros((4, 4), np.float32)
            cv_to_gl[0][0] = 1
            cv_to_gl[1][1] = -1  # Invert the y axis
            cv_to_gl[2][2] = -1  # invert the z axis
            cv_to_gl[3][3] = 1
            # T = cv_to_gl * T


            # solvePnP does camera to marker pose; invert to get marker pose to the camera
            transformation = np.linalg.inv(T)
            return T
