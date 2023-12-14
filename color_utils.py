import mkl
mkl.set_num_threads(1)
import cv2
cv2.setNumThreads(1)
import numpy as np


def convert_fast(frame, type_number):
    import argparse
    import os

    import numpy as np
    from PIL import Image
    import cv2

    def load_lms(img):
        assert isinstance(img, np.ndarray), "Image must be a numpy array"
        assert img.shape[2] == 3, "Image must be a 3 channel RGB image"
        img = img/255.0
        assert np.max(img) <= 1.0, "Image values must be in range 0 - 1"
        assert np.min(img) >= 0.0, "Image values must be in range 0 - 1"

        img_lms = np.dot(img[:, :, :3], rgb_to_lms())

        return img_lms

    def rgb_to_lms():
        """
        Matrix for RGB color-space to LMS color-space transformation.
        """
        return np.array([[17.8824, 43.5161, 4.11935],
                         [3.45565, 27.1554, 3.86714],
                         [0.0299566, 0.184309, 1.46709]]).T

    def lms_to_rgb() -> np.ndarray:
        """
        Matrix for LMS colorspace to RGB colorspace transformation.
        """
        return np.array([[0.0809, -0.1305, 0.1167],
                         [-0.0102, 0.0540, -0.1136],
                         [-0.0004, -0.0041, 0.6935]]).T
    def lms_protanopia_sim(degree: float = 0.5) -> np.ndarray:
        """
        Matrix for Simulating Protanopia colorblindness from LMS color-space.
        :param degree: Protanopia degree.
        """
        degree_p = degree
        degree_d = 1.0 - degree_p
        return np.array([[1 - degree_p, 2.02344 * degree_p, -2.52581 * degree_p],
                         [0.494207 * degree_d, 1 - degree_d, 1.24827 * degree_d],
                         [0, 0, 1]]).T

    frame = frame.copy()

    # Load the image file in LMS colorspace
    img_lms = load_lms(frame)

    transform = lms_protanopia_sim()

    # Transforming the LMS Image
    img_sim = np.dot(img_lms, transform)

    # Converting back to RGB colorspace
    img_sim = np.uint8(np.dot(img_sim, lms_to_rgb()) * 255)

    return img_sim




def convert(frame, type_number):
    # !/usr/bin/env python

    # import webcolors
    import numpy as np
    import os
    import sys
    # from texttable import Texttable

    class ColorBlindConverter(object):
        def __init__(self):
            (
                self.Normal,
                self.Protanopia,
                self.Deuteranopia,
                self.Tritanopia,
                self.Protanomaly,
                self.Deuteranomaly,
                self.Tritanomaly,
                self.Monochromacy
            ) = range(8)

            self.powGammaLookup = np.power(np.linspace(0, 256, 256) / 256, 2.2)
            self.conversion_coeffs = [
                {'cpu': 0.735, 'cpv': 0.265, 'am': 1.273463, 'ayi': -0.073894},
                {'cpu': 1.140, 'cpv': -0.140, 'am': 0.968437, 'ayi': 0.003331},
                {'cpu': 0.171, 'cpv': -0.003, 'am': 0.062921, 'ayi': 0.292119}]

        def _inversePow(self, x):
            return int(255.0 * float(0 if x <= 0 else (1 if x >= 1 else np.power(x, 1 / 2.2))))

        def convert(self, rgb, cb_type):
            self.rgb = rgb
            self.cb_type = cb_type

            if self.cb_type == 0:
                self.converted_rgb = self._convert_normal()
            elif self.cb_type in range(1, 4):
                self.converted_rgb = self._convert_colorblind()
            elif self.cb_type in range(4, 7):
                self.converted_rgb = self._convert_anomylize(self._convert_colorblind())
            elif self.cb_type == 7:
                self.converted_rgb = self._convert_monochrome()
            return self.converted_rgb

        def _convert_normal(self):
            return self.rgb

        def _convert_colorblind(self):

            wx = 0.312713;
            wy = 0.329016;
            wz = 0.358271;

            cpu, cpv, am, ayi = self.conversion_coeffs[{
                1: 0, 4: 0,
                2: 1, 5: 1,
                3: 2, 6: 2,
            }[self.cb_type]].values()

            r, g, b = self.rgb

            cr = self.powGammaLookup[r]
            cg = self.powGammaLookup[g]
            cb = self.powGammaLookup[b]

            # rgb -> xyz
            cx = (0.430574 * cr + 0.341550 * cg + 0.178325 * cb)
            cy = (0.222015 * cr + 0.706655 * cg + 0.071330 * cb)
            cz = (0.020183 * cr + 0.129553 * cg + 0.939180 * cb)

            sum_xyz = cx + cy + cz
            cu = 0
            cv = 0

            if (sum_xyz != 0):
                cu = cx / sum_xyz
                cv = cy / sum_xyz

            nx = wx * cy / wy
            nz = wz * cy / wy
            clm = 0
            dy = 0

            clm = (cpv - cv) / (cpu - cu) if (cu < cpu) else (cv - cpv) / (cu - cpu)
            clyi = cv - cu * clm
            du = (ayi - clyi) / (clm - am)
            dv = (clm * du) + clyi

            sx = du * cy / dv
            sy = cy
            sz = (1 - (du + dv)) * cy / dv

            # xyz->rgb
            sr = (3.063218 * sx - 1.393325 * sy - 0.475802 * sz)
            sg = (-0.969243 * sx + 1.875966 * sy + 0.041555 * sz)
            sb = (0.067871 * sx - 0.228834 * sy + 1.069251 * sz)

            dx = nx - sx
            dz = nz - sz

            # xyz->rgb

            dr = (3.063218 * dx - 1.393325 * dy - 0.475802 * dz)
            dg = (-0.969243 * dx + 1.875966 * dy + 0.041555 * dz)
            db = (0.067871 * dx - 0.228834 * dy + 1.069251 * dz)

            adjr = ((0 if sr < 0 else 1) - sr) / dr if dr > 0 else 0
            adjg = ((0 if sg < 0 else 1) - sg) / dg if dg > 0 else 0
            adjb = ((0 if sb < 0 else 1) - sb) / db if db > 0 else 0

            adjust = max([
                0 if (adjr > 1 or adjr < 0) else adjr,
                0 if (adjg > 1 or adjg < 0) else adjg,
                0 if (adjb > 1 or adjb < 0) else adjb])

            sr = sr + (adjust * dr)
            sg = sg + (adjust * dg)
            sb = sb + (adjust * db)

            return [self._inversePow(sr), self._inversePow(sg), self._inversePow(sb)]

        def _convert_anomylize(self, p_cb):
            v = 1.75
            d = v + 1

            r_orig, g_orig, b_orig = self.rgb
            r_cb, g_cb, b_cb = p_cb

            r_new = (v * r_cb + r_orig) / d
            g_new = (v * g_cb + g_orig) / d
            b_new = (v * b_cb + b_orig) / d

            return [int(r_new), int(g_new), int(b_new)]

        def _convert_monochrome(self):
            r_old, g_old, b_old = self.rgb
            g_new = (r_old * 0.299) + (g_old * 0.587) + (b_old * 0.114)
            return [int(g_new)] * 3

    cb_types = {
        'Normal': '(normal vision)',
        'Protanopia': '(red-blind)',
        'Deuteranopia': '(green-blind)',
        'Tritanopia': '(blue-blind)',
        'Protanomaly': '(red-weak)',
        'Deuteranomaly': '(green-weak)',
        'Tritanomaly': '(blue-weak)',
        'Monochromacy': '(totally colorblind)'
    }
    cbconv = ColorBlindConverter()
    shape = frame.shape
    frame = frame.reshape((-1, 3))
    new_frame = frame.copy()
    for i in range(frame.shape[0]):
        new_frame[i] = cbconv.convert(frame[i], type_number)
    frame = new_frame.reshape(shape)
    return frame


