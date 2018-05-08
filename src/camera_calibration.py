"""This file defines the CameraCalibration class."""
import numpy as np
import re


class CameraCalibration(object):
    """The camera described by its intrinsic and distorsion matrices."""

    def __init__(self, debug=False):
        """Initialize a new CameraCalibration object."""
        self.m_intrinsic = self.matrix_from_file("Intrinsic.txt")
        self.m_distorsion = self.matrix_from_file("Distortion.txt")

        if debug:
            print("==========================================================")
            print("CameraCalibration initialized.")
            print("Camera Matrix:")
            print(self.m_intrinsic)
            print("Distortion Coeffitient:")
            print(self.m_distorsion)
            print("==========================================================")

    @staticmethod
    def matrix_from_file(fname):
        """Read a NumPy matrix from a text file."""
        with open(fname) as f:
            cont = f.readlines()
            m = [[float(v) for v in re.sub('  ', ' ', l).split(' ')]
                 for l in cont]
            return np.array(m)

    def get_matrix34(self):
        """Return the camera calibration as a 3x4 NumPy matrix."""
        cparam = np.zeros((4, 4))  # Why 4x4?!

        for j in range(3):
            for i in range(3):
                cparam[i][j] = self.m_intrinsic[i][j]
        for i in range(4):
            cparam[3][i] = self.m_distorsion[i]

        return cparam
