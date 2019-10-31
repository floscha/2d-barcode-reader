"""This file defines the Marker class."""
import cv2
import numpy as np


class Marker(object):
    """A 2D barcode marker."""

    def __init__(self, points):
        """Initialize a new Marker object."""
        self.id = -1
        self.points = np.array(points, np.float32)
        self.r = None
        self.t = None
        self.transformation = None

    @staticmethod
    def rotate(im):
        """Rotate the given image for 90 degrees."""
        out = im.copy()
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                out[i, j] = im[im.shape[1]-j-1, i]
        return out

    @staticmethod
    def hamm_dist_marker(bits):
        ids = [[1, 0, 0, 0, 0],
               [1, 0, 1, 1, 1],
               [0, 1, 0, 0, 1],
               [0, 1, 1, 1, 0]]

        dist = 0

        for y in range(5):
            min_sum = 1e5  # hamming distance to each possible word

            for p in range(4):
                sum_ = 0
                for x in range(5):
                    sum_ += bits[y, x] == 0 if ids[p][x] else 1

                if min_sum > sum_:
                    min_sum = sum_

            #do the and
            dist += min_sum

        return dist

    @staticmethod
    def mat2id(bits):
        val = 0
        for y in range(5):
            val <<= 1
            if [y, 1]:
                val |= 1
            val <<= 1
            if [y, 3]:
                val |= 1

        return val

    def get_marker_id(self, marker_image):
        _, th = cv2.threshold(marker_image,
                              thresh=0,
                              maxval=255,
                              type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # debug
        # cv2.imshow("Marker", cv2.resize(marker_image, (200,200)))

        cell_size = marker_image.shape[0] // 7

        # check for black border
        for y in range(7):
            inc = 1 if (y == 0 or y == 6) else 6
            x = 0
            while x < 7:
                cell_x = x * cell_size
                cell_y = y * cell_size
                cell = th[cell_x:cell_x+cell_size, cell_y:cell_y+cell_size]

                nz = cv2.countNonZero(cell)
                if nz > cell_size ** 2 / 2:
                    return
                x += inc

        # identification of the marker code
        bit_matrix = np.zeros((5, 5))

        for y in range(5):
            for x in range(5):
                cell_x = (x+1) * cell_size
                cell_y = (y+1) * cell_size
                cell = th[cell_x:cell_x+cell_size, cell_y:cell_y+cell_size]

                nz = cv2.countNonZero(cell)
                if nz > cell_size ** 2 / 2:
                    bit_matrix[y, x] = 1

        # check all possible rotations
        rotations = []
        distances = []

        rotations.append(bit_matrix)
        distances.append(self.hamm_dist_marker(rotations[0]))

        min_dist = [distances[0], 0]
        for i in range(4):
            # get the hamming distance to the nearest possible word
            rotations.append(self.rotate(rotations[i-1]))
            distances.append(self.hamm_dist_marker(rotations[i]))

            if distances[i] < min_dist[0]:
                min_dist[0] = distances[i]
                min_dist[1] = i

        if min_dist[0] == 0:
            return self.mat2id(rotations[min_dist[1]])
