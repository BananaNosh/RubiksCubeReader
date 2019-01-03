import imutils
from imutils.perspective import order_points
import cv2
import numpy as np
import math
import argparse
from threading import Thread

color_spaces_hsv = [
    [
        ((100, 131, 0), (125, 255, 255)),  # blue
        ((25, 0, 0), (90, 255, 190)),  # green
        ((15, 0, 165), (90, 100, 255)),  # yellow
        ((85, 0, 165), (185, 105, 255)),  # white
        ((0, 0, 165), (15, 255, 255)),  # orange
        [((0, 95, 43), (13, 255, 156)), ((150, 95, 43), (255, 255, 156))]  # red
    ],
    [
        ((60, 0, 73), (170, 255, 255)),  # blue
        ((25, 0, 0), (90, 255, 190)),  # green
        ((15, 0, 165), (90, 100, 255)),  # yellow
        ((85, 0, 165), (185, 105, 255)),  # white
        ((0, 0, 165), (15, 255, 255)),  # orange
        ((0, 110, 49), (190, 255, 161))  # red]
    ]
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
    for j, bounds in enumerate(color_spaces_hsv[0]):
        if type(bounds) is not list:
            bounds = [bounds]

        accum_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in bounds:
            mask = cv2.inRange(hsv, lower, upper)
            accum_mask = cv2.bitwise_or(accum_mask, mask)
        if accum_mask:
            return j
    return 6


def find_colored_squares_in_image(image, colors_to_find=9):
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 0, 40)

    rectangles = get_recs(edged, colored_image=image)

    rectangles = list(rectangles)

    if len(rectangles) < colors_to_find:
        for i, (lower, upper) in enumerate(zip([np.array([0, 0, 70]), np.array([100, 0, 0])],
                                               [np.array([91, 255, 170]), np.array([169, 255, 42])])):
            only_colored = cv2.inRange(image, lower, upper)
            colored_recs = get_recs(only_colored, rel_similarity_threshold=0.2, already_found_recs=rectangles)
            if len(colored_recs):
                rectangles.extend(colored_recs)
                rectangles = list(remove_doubles(rectangles))

    if len(rectangles) == 0:
        return [], image

    if len(rectangles) > colors_to_find:
        print("Too many squares found")
        return [], image


    image_with_recs = image.copy()
    recs_reshaped = np.array([order_points(np.reshape(rec, (4, 2))).astype(np.int32) for rec in rectangles])
    mean_height = int(np.max(recs_reshaped[:, [3, 2], 1] - recs_reshaped[:, [0, 1], 1]))
    recs_reshaped = sorted(recs_reshaped, key=lambda x: x[0, 0] + mean_height * 5 * (x[0, 1] // mean_height))
    found_colors = []
    for i, rec in enumerate(recs_reshaped):
        x = rec[0, 0]
        y = rec[0, 1]
        color = color_for_rect(image, rec)
        cv2.putText(image_with_recs, f"{color}", (x + 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255])
        found_colors.append(color)

    # rectangles = [rec.reshape((4, 1, 2)) for rec in recs_reshaped]
    cv2.drawContours(image_with_recs, rectangles, -1, (0, 255, 0), 2)
    return found_colors, image_with_recs


def find_colored_squares_in_image_with_colors(image, colors_to_find=9):
    image = imutils.resize(image, height=500)
    # image = scale_image_lighting(image, {"type": "scb", "satlevel": 0.01})
    rectangles = []
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for bounds in color_spaces_hsv[0]:
        if type(bounds) is not list:
            bounds = [bounds]
        only_color = np.zeros(image.shape[:2], dtype=np.uint8)
        for lower, upper in bounds:
            mask = cv2.inRange(hsv_image, lower, upper)
            only_color = cv2.bitwise_or(only_color, mask)
        colored_recs = get_recs(only_color, rel_similarity_threshold=0.2, colored_image= image, already_found_recs=rectangles)
        if len(colored_recs):
            rectangles.extend(colored_recs)
    rectangles = list(remove_doubles(rectangles))

    if len(rectangles) > colors_to_find:
        print("Too many squares found")
        return [], image

    if len(rectangles) == 0:
        return [], image

    image_with_recs = image.copy()

    recs_reshaped = np.array([order_points(np.reshape(rec, (4, 2))).astype(np.int32) for rec in rectangles])
    mean_height = int(np.mean(recs_reshaped[:, [3, 2], 1] - recs_reshaped[:, [0, 1], 1]))
    min_y = np.min(recs_reshaped[:, [0, 1], 1])
    recs_reshaped = sorted(recs_reshaped, key=lambda x: x[0, 0] + mean_height * 5 * int((x[0, 1] - min_y) // mean_height + 0.25))
    found_colors = []
    for i, rec in enumerate(recs_reshaped): # TODO make simpler => colors already given above
        x = rec[0, 0]
        y = rec[0, 1]
        color = color_for_rect(image, rec)
        cv2.putText(image_with_recs, f"{color}", (x + 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        found_colors.append(color)

    # rectangles = [rec.reshape((4, 1, 2)) for rec in recs_reshaped]
    cv2.drawContours(image_with_recs, rectangles, -1, (0, 255, 0), 2)
    return found_colors, image_with_recs


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
    cnts = cv2.findContours(image_to_extract_from, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rectangles = []
    if already_found_recs is not None:
        for rec in already_found_recs:
            rectangles.append(np.array([rec, cv2.contourArea(rec)]))
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found a square
        if len(approx) == 4:
            sq_area = cv2.contourArea(approx)
            if sq_area < 500:
                break

            # check for being filled
            if np.mean(get_center_rec(approx, image_to_extract_from)) < 230:
                continue

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


def colors_from_video(video_path=None, show=False, colors_to_find=9, mid=4):
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

        found_colors, frame = find_colored_squares_in_image_with_colors(frame, colors_to_find=colors_to_find)

        if frames_after_sixth_side > 0:
            frames_after_sixth_side += 1
        mid_color = found_colors[mid] if len(found_colors) > mid else -1
        if len(found_colors) == colors_to_find and mid_color < 6:
            colors_for_sides[mid_color].append(found_colors)
            if len(colors_for_sides[mid_color]) > 50:
                colors_for_sides[mid_color].pop(0)
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


def all_possible_color_areas(image, window_size=None):
    original_image = image
    if window_size is None:
        window_size = image.shape[0] // 10
    found_points = points_with_variation_in_border(image, window_size, 300)
    for x, y in found_points:
        border = image.shape[0] // 36
        cv2.rectangle(image, (y + border, x + border), (y + window_size - border, x + window_size - border), (0, 0, 255), cv2.FILLED)
        # cv2.circle(image, (y + window_size//2, x + window_size//2), 1, (0, 0, 255), 1)
    # cv2.imshow("im", image)
    edged = cv2.inRange(image, (0, 0, 255), (0, 0, 255))
    # cv2.imshow("bin", edged)
    cnts = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rectangles = []
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        epsilon_factor = 0.03
        approx = cv2.approxPolyDP(c, epsilon_factor * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found a square
        if len(approx) == 4:
            rectangles.append(approx)
    centers = [np.mean(r, axis=(0, 1), dtype=np.int32) for r in rectangles]
    dists = []
    for i, center1 in enumerate(centers):
        cv2.putText(original_image, str(i), tuple(center1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.circle(original_image, tuple(center1), 1, (255, 0, 0), 1)
        for j, center2 in enumerate(centers[i+1:]):
            j = j + i + 1
            dists.append((np.sum(np.square(center2 - center1)), (i, j)))
    dists = sorted(dists)
    print("dists", dists)
    dists_dict = dict((i, [i]) for i in range(len(centers)))
    center_index = None
    min_in_range_of_center = 7
    for dist, (i, j) in dists:
        dists_dict[i].append(j)
        dists_dict[j].append(i)
        if center_index is None:
            if len(dists_dict[i]) > min_in_range_of_center:
                center_index = i
            elif len(dists_dict[j]) > min_in_range_of_center:
                center_index = j
    print("centerIndex", center_index, dists_dict[center_index])
    cv2.drawContours(original_image, rectangles, -1, (0, 255, 0))
    cv2.imshow("cnts", original_image)
    cv2.waitKey(0)


def points_with_variation_in_border(image, window_size, variation_threshold=400):
    if window_size > min(image.shape[:2]):
        raise AssertionError("Window bigger than image")
    window = image[:window_size, :window_size]
    last_color_sum = np.sum(window, axis=(0, 1))
    last_squared_sum = np.sum(np.square(window, dtype=np.int32), axis=(0, 1))
    first_in_row_sum = last_color_sum.copy()
    first_in_row_squared = last_squared_sum.copy()
    n = window_size ** 2
    found_points = []
    for x in range(1, image.shape[0] - window_size):
        first_in_row_sum = first_in_row_sum - np.sum(image[x - 1, 0:window_size], axis=0) \
                           + np.sum(image[x - 1 + window_size, 0:window_size], axis=0)
        first_in_row_squared = first_in_row_squared - np.sum(np.square(image[x - 1, 0:window_size], dtype=np.int32),
                                                             axis=0) \
                               + np.sum(np.square(image[x - 1 + window_size, 0:window_size], dtype=np.int32), axis=0)
        last_color_sum = first_in_row_sum.copy()
        last_squared_sum = first_in_row_squared.copy()
        for y in range(1, image.shape[1] - window_size):
            last_color_sum = last_color_sum - np.sum(image[x:x + window_size, y - 1], axis=0) \
                             + np.sum(image[x:x + window_size, y - 1 + window_size], axis=0)
            last_squared_sum = last_squared_sum - np.sum(np.square(image[x:x + window_size, y - 1], dtype=np.int32),
                                                         axis=0) \
                               + np.sum(np.square(image[x:x + window_size, y - 1 + window_size], dtype=np.int32),
                                        axis=0)
            std = np.mean(last_squared_sum / n - (last_color_sum / n) ** 2)
            if std < variation_threshold:
                found_points.append((x, y))
            # print(f"pos: ({x},{y}) - {std}")
    return found_points


class CubeWebcamStream:
    def __init__(self, src=0, name="CubeVideo", colors_to_find=9):
        self.stream = cv2.VideoCapture(src)
        self.colors_to_find = colors_to_find
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
            found_colors, frame = find_colored_squares_in_image_with_colors(raw_frame, colors_to_find=self.colors_to_find)
            if len(found_colors) == self.colors_to_find:
                self.frame = frame
                self.colors = found_colors
            else:
                self.frame = raw_frame if frame is None else frame
                self.colors = None

    def read(self):
        # return the frame most recently read
        return self.colors, self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


if __name__ == '__main__':
    # image_names = [f"cube_1_{i}.png" for i in range(6)]
    image_names = [f"cube_1_5_warped.png"]
    for name in image_names:
        image = cv2.imread(f"./data/{name}")
        image = imutils.resize(image, image.shape[0] // 1)
        new_image = image.copy()
        indices = np.where(np.sum(np.square((image.T - np.mean(image, axis=2).T).T), axis=2) < 100)
        new_image[indices] = 255
        # cv2.imshow(f"image-{name}", image)
        # cv2.imshow(f"new-{name}", new_image)
        # image[np.where()]
        all_possible_color_areas(new_image)
    cv2.waitKey(0)
    print("ready")
    #
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", help="path to input image")
    # ap.add_argument("-v", "--video", help="path to input video")
    # ap.add_argument("-c", "--count", type=int, help="number of field per side")
    #
    # args = vars(ap.parse_args())
    #
    # image_path = args.get("image", False)
    # video_path = args.get("video", None)
    # fields_count = args.get("count", None)
    # fields_count = 9 if fields_count is None else fields_count
    # if image_path:
    #     _image = cv2.imread(image_path)
    #     _found_colors, _image = find_colored_squares_in_image_with_colors(_image, fields_count)
    #     print(f"colors {image_path}:", _found_colors)
    #     cv2.imshow("Output", _image)
    #     cv2.waitKey(0)
    # elif video_path:
    #     colors = colors_from_video(video_path=video_path, show=False)
    #     print(colors)
    # else:
    #     stream = CubeWebcamStream(colors_to_find=fields_count).start()
    #     while True:
    #         colors, frame = stream.read()
    #
    #         cv2.imshow("Frame", frame)
    #         key = cv2.waitKey(1) & 0xFF
    #         if colors is not None:
    #             print("colors", colors)
    #
    #         if key == ord("q"):
    #             break
    #     stream.stop()
    #     cv2.destroyAllWindows()
