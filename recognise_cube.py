import imutils
from imutils.perspective import order_points
import cv2
import numpy as np
import math
import argparse
from threading import Thread

color_spaces_hsv = [
    ((100, 131, 0), (125, 255, 255)),  # blue
    ((70, 0, 0), (90, 255, 190)),  # green
    ((40, 0, 165), (90, 100, 255)),  # yellow
    ((85, 0, 165), (185, 105, 255)),  # white
    ((0, 0, 165), (15, 255, 255)),  # orange
    ((0, 110, 49), (190, 255, 161))  # red
]


def transform_according_to_reference_square(image, reference_points):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(reference_points)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    new_ref_width_1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    new_ref_width_2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    new_ref_width = max(int(new_ref_width_1), int(new_ref_width_2))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    new_ref_height_1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    new_ref_height_2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    new_ref_height = max(int(new_ref_height_1), int(new_ref_height_2))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        rect[0],
        rect[0] + [new_ref_width - 1, 0],
        rect[0] + [new_ref_width - 1, new_ref_height - 1],
        rect[0] + [0, new_ref_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (image.shape[1], image.shape[0]))

    # return the warped image
    return warped


def color_for_rect(image, rect):
    center_rec = get_center_rec(rect, image, rel_margin=0.25)
    mean_color = np.mean(center_rec, axis=(0, 1))
    hsv = cv2.cvtColor(mean_color.reshape((1, 1, 3)).astype(np.uint8), cv2.COLOR_BGR2HSV)
    for j, (lower, upper) in enumerate(color_spaces_hsv):
        if cv2.inRange(hsv, lower, upper):
            return j
    return 6


def find_colored_squares_in_image(image):
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 100)
    # print("STEP 1: Edge Detection")

    rectangles = get_recs(edged, colored_image=image)
    if len(rectangles) == 0:
        return [], image

    rectangles = list(rectangles)

    if len(rectangles) < 9:
        for i, (lower, upper) in enumerate(zip([np.array([0, 0, 70]), np.array([100, 0, 0])],
                                               [np.array([91, 255, 170]), np.array([169, 255, 42])])):
            only_colored = cv2.inRange(image, lower, upper)
            colored_recs = get_recs(only_colored, rel_similarity_threshold=0.2, already_found_recs=rectangles)
            if len(colored_recs):
                rectangles.extend(colored_recs)
                rectangles = list(remove_doubles(rectangles))

    if len(rectangles) > 9:
        print("Too many squares found")
        return [], image

    recs_reshaped = np.array([order_points(np.reshape(rec, (4, 2))).astype(np.int32) for rec in rectangles])
    mean_height = int(np.max(recs_reshaped[:, [3, 2], 1] - recs_reshaped[:, [0, 1], 1]))
    recs_reshaped = sorted(recs_reshaped, key=lambda x: x[0, 0] + mean_height * 5 * (x[0, 1] // mean_height))
    found_colors = []
    for i, rec in enumerate(recs_reshaped):
        x = rec[0, 0]
        y = rec[0, 1]
        color = color_for_rect(image, rec)
        cv2.putText(image, f"{color}", (x + 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255])
        found_colors.append(color)

    # rectangles = [rec.reshape((4, 1, 2)) for rec in recs_reshaped]
    cv2.drawContours(image, rectangles, -1, (0, 255, 0), 2)
    return found_colors, image


def get_warp_ratios(rectangles):
    np_rectangles = np.zeros((len(rectangles), 4, 1, 2))
    for i, rect in enumerate(rectangles):
        np_rectangles[i] = rect
    # print(np_rectangles.shape, rectangles.shape)
    flattened_recs = np_rectangles[:, :, 0]
    right_order_recs = np.array([order_points(rec) for rec in flattened_recs])
    warp_ratio_dist = right_order_recs[:, 3] - right_order_recs[:, 0]
    warp_ratios = np.abs(warp_ratio_dist[:, 0] / warp_ratio_dist[:, 1])
    return warp_ratios


def get_recs(image_to_extract_from, rel_similarity_threshold=0.05, colored_image=None, already_found_recs=None):
    cnts = cv2.findContours(image_to_extract_from.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rectangles = []
    if already_found_recs is not None:
        for rec in already_found_recs:
            rectangles.append(np.array([rec, cv2.contourArea(rec)]))
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found a square
        if len(approx) == 4:
            sq_area = cv2.contourArea(approx)
            if sq_area < 1000:
                break

            # check for similarity in color within approx
            if colored_image is not None:
                image_part = get_center_rec(approx, colored_image)
                mean_color_std = np.mean(np.std(image_part, axis=(0, 1)))
                if mean_color_std > 15:
                    continue

            # check for similarity in size
            if len(rectangles) > 0 \
                    and not math.isclose(sq_area, rectangles[-1][1], rel_tol=rel_similarity_threshold):
                if len(rectangles) < 3:
                    rectangles = []
                else:
                    continue
            rectangles.append(np.array([approx, sq_area]))
    if len(rectangles) == 0:
        return np.array([])
    rectangles = np.array(rectangles)
    rectangles = rectangles[:, 0]
    # remove doubles
    rectangles = sorted(rectangles, key=lambda x: np.min(np.sum(x[:, 0], axis=1)))
    rectangles = remove_doubles(rectangles)
    return rectangles


def get_center_rec(approx, image, rel_margin=0.1):
    reshaped = np.reshape(approx, (4, 2))
    x_min, y_min = np.min(reshaped, axis=0)
    x_max, y_max = np.max(reshaped, axis=0)
    width = x_max - x_min
    height = y_max - y_min
    x_min = int(x_min + rel_margin * width)
    x_max = int(x_max - rel_margin * width)
    y_min = int(y_min + rel_margin * height)
    y_max = int(y_max - rel_margin * height)
    image_part = image[y_min:y_max, x_min:x_max]
    return image_part


def remove_doubles(rectangles):
    for i, rec in enumerate(rectangles[:-1]):
        for neighbour in rectangles[i + 1:]:
            if np.sum(np.abs(rec[np.argmin(np.sum(rec[:, 0], axis=1)), 0]
                             - neighbour[np.argmin(np.sum(neighbour[:, 0], axis=1)), 0])) < 10:
                rectangles[i] = None
                break
    rectangles = np.array([rec for rec in rectangles if rec is not None])
    return rectangles


def colors_from_video(video_path=None, show=False):
    colors_for_sides = [[] for _ in range(6)]
    frames_after_sixth_side = 0
    if not video_path:
        vc = cv2.VideoCapture(0)
    else:
        vc = cv2.VideoCapture(video_path)
    while True:
        _, frame = vc.read()
        if frame is None or (not video_path and frames_after_sixth_side == 10):
            break

        found_colors, frame = find_colored_squares_in_image(frame)

        if frames_after_sixth_side > 0:
            frames_after_sixth_side += 1
        if len(found_colors) == 9 and found_colors[4] < 6:
            colors_for_sides[found_colors[4]].append(found_colors)
            if len(colors_for_sides[found_colors[4]]) > 50:
                colors_for_sides[found_colors[4]].pop(0)
            if frames_after_sixth_side == 0 and not any([len(side) == 0 for side in colors_for_sides]):
                frames_after_sixth_side = 1

        if show:
            cv2.imshow("Frame", frame)
            frame_rate = vc.get(cv2.CAP_PROP_FPS)
            time_per_frame = int(1000 / frame_rate)
            key = cv2.waitKey(time_per_frame if show else 1) & 0xFF
            if key == ord("q"):
                break
    if show:
        cv2.destroyAllWindows()
    for i, side in enumerate(colors_for_sides):
        side = np.array(side).transpose()
        colors_for_sides[i] = []
        for field in side:
            numbers, counts = np.unique(field, return_counts=True)
            number = numbers[np.argmax(counts)]
            colors_for_sides[i].append(number)
    vc.release()
    return colors_for_sides


class CubeWebcamStream:
    def __init__(self, src=0, name="CubeVideo"):
        self.stream = cv2.VideoCapture(src)
        _, self.frame = self.stream.read()
        self.colors = None
        self.name = name
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            _, raw_frame = self.stream.read()
            found_colors, frame = find_colored_squares_in_image(raw_frame)
            if len(found_colors) == 9:
                self.frame = frame
                self.colors = found_colors
            else:
                self.frame = raw_frame
                self.colors = None

    def read(self):
        # return the frame most recently read
        return self.colors, self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to input image")
    ap.add_argument("-v", "--video", help="path to input video")

    args = vars(ap.parse_args())

    image_path = args.get("image", False)
    video_path = args.get("video", None)
    if image_path:
        _image = cv2.imread(image_path)
        _found_colors, _image = find_colored_squares_in_image(_image)
        print(f"colors {image_path}:", _found_colors)
        cv2.imshow("Output", _image)
        cv2.waitKey(0)
    elif video_path:
        colors = colors_from_video(video_path=video_path, show=not video_path)
        print(colors)
    else:
        stream = CubeWebcamStream().start()
        while True:
            colors, frame = stream.read()

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            print("colors", colors)

            if key == ord("q"):
                break
        cv2.destroyAllWindows()
