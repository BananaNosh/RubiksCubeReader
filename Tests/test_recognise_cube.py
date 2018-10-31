import unittest
from recognise_cube import colors_from_video, find_colored_squares_in_image
import cv2


class RecogniseCubeTest(unittest.TestCase):

    def test_from_video(self):
        colors = colors_from_video("../data/cube.avi")
        self.assertEqual(6, len(colors))
        expected = [[3, 3, 2, 0, 0, 1, 3, 0, 0], [4, 1, 2, 4, 1, 2, 1, 1, 1], [2, 5, 5, 0, 2, 4, 2, 2, 0],
                    [5, 4, 4, 1, 3, 3, 3, 3, 1], [0, 4, 0, 2, 4, 0, 4, 5, 4], [5, 5, 1, 5, 5, 3, 3, 2, 5]]
        for i, cs in enumerate(colors):
            self.assertEqual(9, len(cs))
            self.assertEqual(expected[i], cs)

    def test_from_image(self):
        image = cv2.imread("../data/cube_2x2_0.jpg")

        colors, _ = find_colored_squares_in_image(image, colors_to_find=4)
        expected = [6, 0, 2, 5]
        self.assertEqual(expected, colors)