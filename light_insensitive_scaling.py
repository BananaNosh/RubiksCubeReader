# Build after the matlab implementation by Jason Su: https://web.stanford.edu/~sujason/ColorBalancing/Code/
import numpy as np
import cv2
import argparse


CAT_TYPE_BRADFORD = 0
CAT_TYPE_KRIES = 1
CAT_TYPE_XYZ_SCALING = 2

XYZ_D65 = np.array([95.04, 100, 108.88], dtype=np.float32)
RGB2YUV = np.array([[0.299, 0.587, 0.114],
                    [-0.299, -0.587, 0.886],
                    [0.701, -0.587, -0.114]])
YUV2RGB = np.linalg.inv(RGB2YUV)
RGB2XYZ = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
]
XYZ2RGB = np.linalg.inv(RGB2XYZ)


def cb_cat_matrix(source_white, dest_white, type=CAT_TYPE_BRADFORD):
    if type == CAT_TYPE_BRADFORD:
        mat = np.array([[0.8951000,  0.2664000, -0.1614000],
                        [-0.7502000,  1.7135000,  0.0367000],
                        [0.0389000, -0.0685000,  1.029600]])
        # inv_mat = np.array([[0.9869929, -0.1470543, 0.1599627],
        #                 [0.4323053, 0.5183603, 0.0492912],
        #                 [-0.0085287, 0.0400428, 0.9684867]])
    elif type == CAT_TYPE_KRIES:
        mat = np.array([[0.4002400, 0.7076000, -0.0808100],
                        [-0.2263000, 1.1653200, 0.0457000],
                        [0.0000000, 0.0000000, 0.9182200]])
        # inv_mat = np.array([[1.8599364, -1.1293816, 0.2198974],
        #                     [0.3611914, 0.6388125, -0.0000064],
        #                     [0, 0, 1.0890636]])
    elif type == CAT_TYPE_XYZ_SCALING:
        mat = np.identity(3)
    else:
        raise ValueError("Not yet implemented")

    # if dest_white is None:
    #     dest_white = (255, 255, 255)
    # if source_white is None:
    #     # most_white_pixel_index = np.argmin(np.sum(np.square(np.subtract(255, image, dtype=np.int32)), axis=2))
    #     # source_white = image[np.unravel_index(most_white_pixel_index, image.shape[0:2])]
    #     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     most_white_pixel_index = np.argmin(hsv_image[:, :, 1] + hsv_image[:, :, 2])
    #     source_white = image[np.unravel_index(most_white_pixel_index, image.shape[0:2])]

    # source_white = np.array(source_white, dtype=np.float32)
    # dest_white = np.array(dest_white, dtype=np.float32)
    # dest_white1 = np.matmul(dest_white[::-1] / 255, rgb2_xyz)
    # source_white1 = np.matmul(source_white[::-1] / 255, rgb2_xyz)
    # print(dest_white1, source_white1)

    # source_white = cv2.cvtColor(np.reshape(source_white / 255, (1, 1, 3)), cv2.COLOR_BGR2XYZ)[0, 0]
    # dest_white = cv2.cvtColor(np.reshape(dest_white / 255, (1, 1, 3)), cv2.COLOR_BGR2XYZ)[0, 0]
    # print(source_white, dest_white)

    scale_mat = (np.matmul(mat, dest_white) / np.matmul(mat, source_white)) * np.identity(3)
    # print(scale_mat)

    complete_mat = np.matmul(np.linalg.lstsq(mat, scale_mat, rcond=-1)[0], mat)
    # print("completeMat\n", complete_mat)

    out_mat = np.matmul(XYZ2RGB, np.matmul(complete_mat, RGB2XYZ))
    # print("outmat\n", out_mat)
    return out_mat

    # out = np.matmul(image / 255, out_mat)
    # out = out / np.max(out) * 255
    # return out.astype(np.uint8)

    # xyz = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2XYZ)
    # flattened = reshape_image(xyz)
    # print("xyz_image\n", flattened[1000:1005])
    # flattened_scaled = np.matmul(flattened, complete_mat)
    # scaled_xyz = unshape_image(flattened_scaled.astype(np.float32), image.shape)
    # scaled = cv2.cvtColor(scaled_xyz, cv2.COLOR_XYZ2BGR)
    # print("scaled\n", reshape_image(scaled)[42600:42605])
    # return scaled.astype(np.int8)


def robust_awb(image, cat_type, dev_thresh=0.3, max_iter=100):
    conv_thresh = 0.001
    improve_thresh = 0.000001

    image_reshaped = reshape_image(image)
    # print(image[2:5, 150])
    # print("image_reshaped\n", image_reshaped[63752:63755])

    u_avgs = []
    v_avgs = []
    # tot_grays = []
    for i in range(max_iter):
        # print(f"iteration{i}")
        # Transpose for multiplying from left instead of right
        yuv = np.matmul(image_reshaped[:, ::-1], RGB2YUV.T)
        # TODO try cv2.cvtColor
        # yuv = np.reshape(cv2.cvtColor(np.reshape(image_reshaped, image.shape), cv2.COLOR_BGR2YUV), (-1, 3))
        # print("yuv\n", yuv[1000:1005], yuv.shape)
        # find gray chromaticity - (|u|+|v|)/y
        chromaticity = (np.abs(yuv[:, 1]) + np.abs(yuv[:, 2])) / yuv[:, 0]
        # print("chroma\n", chromaticity[1000:1005], chromaticity.shape)

        chrom_in_thresh = chromaticity < dev_thresh
        # tot_grays.append(np.sum(chrom_in_thresh))
        if not any(chrom_in_thresh):  #tot_grays[-1] == 0:
            # print("No valid gray pixels found")
            break
        gray_indices = np.where(chrom_in_thresh)[0]
        grays = yuv[gray_indices]
        # print("graysum", sum(chrom_in_thresh))
        # print("grays\n", grays[0:5])

        u_avg = np.mean(grays[:, 1])
        v_avg = np.mean(grays[:, 2])
        # print(u_avg, v_avg)
        u_avgs.append(u_avg)
        v_avgs.append(v_avg)

        if max(abs(u_avg), abs(v_avg)) < conv_thresh:
            # print(f"Converged with u_avg and v_avg < {conv_thresh}")
            break
        elif i >= 1 and np.linalg.norm([u_avg - u_avgs[-2], v_avg - v_avgs[-2]]) < improve_thresh:
            # print("u and v no longer improving")
            break
        rgb_est = np.matmul(YUV2RGB, [100, u_avg, v_avg]) # TODO use cv2.cvtColor
        xyz_est = cv2.cvtColor(np.reshape(rgb_est.astype(np.float32), (1, 1, 3)), cv2.COLOR_RGB2XYZ)[0, 0]
        xyz_est = xyz_est / xyz_est[1] * 100  # norm y to 100 (D65 luminance comparable
        # print("xyzEst\n", xyz_est)
        cb_cat_mat = cb_cat_matrix(xyz_est, XYZ_D65, cat_type)
        image_reshaped = np.matmul(image_reshaped[:, ::-1], cb_cat_mat.T)[:, ::-1]
        # print("scaled\n", image_reshaped[42600:42605])

    return unshape_image(image_reshaped, image.shape).astype(np.uint8)


def gray_world(image, cat_type, max_iter=100):
    conv_thresh = 0.001
    improve_thresh = 0.000001

    image_reshaped = reshape_image(image)

    gray_diffs = []
    for i in range(max_iter):
        rgb_est = np.mean(image_reshaped[:, ::-1].T, axis=1)
        gray_diff = np.linalg.norm([rgb_est[0] - rgb_est[1], rgb_est[0] - rgb_est[2], rgb_est[1] - rgb_est[2]])
        gray_diffs.append(gray_diff)

        if gray_diff < conv_thresh:
            print("Converged. RGB difference vector <", conv_thresh)
            break
        elif i > 0 and abs(gray_diffs[-2] - gray_diff) < improve_thresh:
            print("RGB difference vector no longer improving")
            break
        xyz_est = cv2.cvtColor(np.reshape(rgb_est.astype(np.float32), (1, 1, 3)), cv2.COLOR_RGB2XYZ)[0, 0]
        xyz_est = xyz_est / xyz_est[1] * 100  # norm y to 100 (D65 luminance comparable
        cb_cat_mat = cb_cat_matrix(xyz_est, XYZ_D65, cat_type)
        image_reshaped = np.matmul(image_reshaped[:, ::-1], cb_cat_mat.T)[:, ::-1]

    return unshape_image(image_reshaped, image.shape).astype(np.uint8)


def simplest_color_balance(image, sat_level=0.01):
    image_reshaped = reshape_image(image)
    q = [sat_level / 2, 1 - sat_level/2]

    quantiles = np.quantile(image_reshaped, q, axis=0)
    image_reshaped = np.where(image_reshaped < quantiles[0], quantiles[0],
                              np.where(image_reshaped > quantiles[1], quantiles[1], image_reshaped))

    lowest = np.min(image_reshaped, axis=0)
    highest = np.max(image_reshaped, axis=0)
    image_reshaped = (image_reshaped - lowest) * 255 / (highest - lowest)

    return unshape_image(image_reshaped, image.shape).astype(np.uint8)


def reshape_image(image):
    return np.reshape(np.transpose(image, (1, 0, 2)), (-1, 3))


def unshape_image(image, shape):
    return np.transpose(np.reshape(image, (shape[1], shape[0], shape[2])), (1, 0, 2))


def scale_image_lighting(image, args):
    type = args.get("type")
    if type == "scb":
        method = simplest_color_balance
        sat_level = args.get("satlevel")
        m_args = [sat_level if sat_level is not None else 0.01]
    else:
        method = robust_awb if type == "rwb" else gray_world if type == "gw" else None
        if method is None:
            print("type must be given")
        cat_type = args.get("cattype", "bradford")
        cat_type = CAT_TYPE_BRADFORD if cat_type == "bradford" else CAT_TYPE_KRIES if cat_type == "kries" \
            else CAT_TYPE_XYZ_SCALING
        m_args = [cat_type]
    return method(image, *m_args)


if __name__ == '__main__':
    # image = cv2.imread("./data/cube_4x4_0.jpg")
    # image = cv2.imread("./data/test_dog.png")
    # scaled = simplest_color_balance(image)
    # scaled = gray_world(image, CAT_TYPE_XYZ_SCALING)
    # scaled_brad = gray_world(image, CAT_TYPE_BRADFORD)
    # scaled_kries = gray_world(image, CAT_TYPE_KRIES)
    # scaled = robust_awb(image, CAT_TYPE_XYZ_SCALING)
    # scaled_brad = robust_awb(image, CAT_TYPE_BRADFORD)
    # scaled_kries = robust_awb(image, CAT_TYPE_KRIES)
    # cv2.imwrite("./data/scaled.png", scaled)
    # cv2.imshow("image", image)
    # cv2.imshow("scaled", scaled)
    # cv2.imshow("brad", scaled_brad)
    # cv2.imshow("kries", scaled_kries)
    # cv2.waitKey(0)
    # test_im = image[:5, :10]
    # print(test_im.shape)
    # reshaped = reshape_image(test_im)
    # print(reshaped.shape)
    # unshaped = unshape_image(reshaped, test_im.shape)
    # print(unshaped.shape)
    # print(sum(test_im == unshaped))

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to input image")
    ap.add_argument("-v", "--video", help="path to input video")
    ap.add_argument("-t", "--type", required=True, choices=['scb', 'gw', 'rawb'],
                    help="scaling type: simple color balance, gray world, robust auto-white balance")
    ap.add_argument("-c", "--cattype", choices=["bradford", "kries", "xyz"], help="cattype")
    ap.add_argument("-s", "--satlevel", type=float, help="saturation level for scb")

    args = vars(ap.parse_args())

    type = args.get("type")
    if type == "scb":
        method = simplest_color_balance
        sat_level = args.get("satlevel")
        m_args = [sat_level if sat_level is not None else 0.01]
    else:
        method = robust_awb if type == "rwb" else gray_world
        cat_type = args.get("cattype", "bradford")
        cat_type = CAT_TYPE_BRADFORD if cat_type == "bradford" else CAT_TYPE_KRIES if cat_type == "kries" \
            else CAT_TYPE_XYZ_SCALING
        m_args = [cat_type]

    image_path = args.get("image", False)
    video_path = args.get("video", None)
    fields_count = args.get("count", 9)
    if image_path:
        _image = cv2.imread(image_path)
        _image = method(_image, *m_args)
        cv2.imshow("Output", _image)
        cv2.waitKey(0)
    else:
        vc = cv2.VideoCapture(0 if not video_path else video_path)
        while True:
            _, frame = vc.read()
            if frame is None:
                break
            frame = method(frame, *m_args)
            cv2.imshow("scaled frame", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
