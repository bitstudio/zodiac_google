import cv2
import numpy as np
import os
import math
import imageprocess
from array import array
import json


def radial_kernel(r, t, max_dist, step, dtdr=0.01):

    step = math.sin(step * math.pi / 2)
    if step < 0:
        r_ = r * (-step) + (1 + step) * max_dist / 5
        t_ = t - (max_dist - r_) * dtdr * (1 + step)
    else:
        r_ = r * (step) + (1 - step) * max_dist / 5
        t_ = t + (r_) * dtdr * (1 - step)

    return r_, t_


def morph(points, size, center, step):

    width, height = size

    nx = points[:, 0] - center[0]
    ny = points[:, 1] - center[1]
    r = np.sqrt(nx * nx + ny * ny)
    t = np.arctan2(ny, nx)

    r_, t_ = radial_kernel(r, t, math.sqrt(width * width + height * height), step)

    alpha = abs(step)
    x_ = r_ * np.cos(t_) + center[0] * (alpha) + (1 - alpha) * width * 0.5
    y_ = r_ * np.sin(t_) + center[1] * (alpha) + (1 - alpha) * height * 0.5

    return np.stack([x_, y_], axis=1)


def find_a_center(points):
    centroid = np.mean(points, axis=0)
    cx = centroid[0]

    crosses = []
    for i in range(points.shape[0] - 1):
        if (points[i, 0] < cx and cx <= points[i + 1, 0]) or (points[i, 0] >= cx and cx > points[i + 1, 0]):
            crosses.append((points[i, 1] + points[i + 1, 1]) * 0.5)

    max_value = 0
    max_index = 0
    for i in range(len(crosses) // 2):
        v = abs(crosses[2 * i] - crosses[2 * i + 1])
        if v > max_value:
            max_value = v
            max_index = i

    cy = (crosses[2 * max_index] + crosses[2 * max_index + 1]) * 0.5

    return (cx, cy)


def get_contour(img, size, invert=False):
    processed = imageprocess.process_raw(img, size, 0, 0, None, None)
    contour = imageprocess.extract_contour(processed if not invert else 255 - processed)
    eqi_length = imageprocess.re_contour(contour, 256)
    return np.array(eqi_length)


def gen_target_for_web_download(cwd, size, invert):
    targets = []
    for file in sorted(os.listdir(cwd)):
        print(file)
        full_path = os.path.join(cwd, file)
        img = cv2.imread(full_path)
        targets.append(get_contour(img, size, invert))

    return np.array(targets, dtype=np.float32)


def save_target_for_web_download(save_path, source_dir, size, invert):
    targets = gen_target_for_web_download(source_dir, size, invert)

    with open(os.path.join(save_path, "morph_data", "desc.json"), 'w') as outfile:
        data = []
        data.append({"t": "f",
                     "shape": targets.shape,
                     "scale": size,
                     "path": "morph_data/morph.bin"})
        json.dump(data, outfile)

    with open(os.path.join(save_path, "morph_data", "morph.bin"), 'wb') as output:
        array('f', targets.astype(np.float32).flatten().tolist()).tofile(output)


if __name__ == '__main__':

    img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "templates", "large_template_front", "0.10.0,0,100,100.png")

    raw_target_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "morph_targets")
    img2_path = os.path.join(raw_target_dir, "08.png")

    img = cv2.imread(img_path)
    img2 = cv2.imread(img2_path)

    size = (500, 500)

    points = get_contour(img, size)
    center = find_a_center(points)
    print(center)

    points2 = get_contour(img2, (500, 500), True)

    for i in np.arange(-1.0, 1.0, 0.05):
        morphed_points = morph(points if i < 0 else points2, size, center if i < 0 else (250, 250), i)
        canvas = cv2.fillPoly(np.zeros((size[1], size[0], 3)), [morphed_points.astype(np.int32)], (255, 255, 255), 8)
        cv2.imshow("Gen", canvas)
        cv2.waitKey(30)

    # gen download file
    save_target_for_web_download(os.path.join(os.path.dirname(os.path.realpath(__file__)), "web"), raw_target_dir, size, True)
