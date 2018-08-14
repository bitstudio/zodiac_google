import cv2
import numpy as np
import time
import os
import math


def point_sqrdist(p0, p1):
    return (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2


def point_dist(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def interpolate_point(p0, p1, alpha):
    return (p0[0] * (1 - alpha) + p1[0] * (alpha), p0[1] * (1 - alpha) + p1[1] * (alpha))


def re_contour(contour, size):

    c_p = contour[0][0]
    contour_portions = [0.0]
    for i in range(1, len(contour), 1):
        n_p = contour[i][0]
        dist = point_dist(n_p, c_p)
        contour_portions.append(dist + contour_portions[i - 1])
        c_p = n_p
    n_p = contour[0][0]
    dist = point_dist(n_p, c_p)
    total_length = dist + contour_portions[len(contour) - 1]
    contour_portions.append(total_length)

    out_contour = []
    index = 1
    for i in range(size):
        cs = total_length * float(i) * 0.999 / (size - 1)
        while contour_portions[index] < cs:
            index = index + 1
        alpha = (cs - contour_portions[index - 1]) / (contour_portions[index] - contour_portions[index - 1])
        new_point = interpolate_point(contour[index - 1][0], contour[index % len(contour)][0], alpha)
        out_contour.append(new_point)

    return out_contour


def get_normalized_contour_and_stat(contour):

    sx = 0
    sy = 0
    size = max(len(contour), 10)
    for i in range(size):
        sx = sx + contour[i][0]
        sy = sy + contour[i][1]

    centroid = (float(sx) / size, float(sy) / size)

    out_contour = np.zeros((1, size), dtype=np.float32)
    sl = 0
    sll = 0
    for i in range(size):
        sq = point_sqrdist(contour[i], centroid)
        sqrt = math.sqrt(sq)
        sl = sl + sqrt
        sll = sll + sq
        out_contour[0, i] = sqrt

    bias = float(sl) / size
    variance = float(sll) / size - bias * bias

    out_contour = out_contour / variance

    return out_contour, centroid, bias, variance


def setup(start_contour, end_contour):

    max_size = max(len(start_contour), len(end_contour))

    start_contour = re_contour(start_contour, max_size)
    end_contour = re_contour(end_contour, max_size)

    n_start, c_start, b_start, v_start = get_normalized_contour_and_stat(start_contour)
    n_end, c_end, b_end, v_end = get_normalized_contour_and_stat(end_contour)

    res = cv2.filter2D(n_start, cv2.CV_32F, n_end, anchor=(0, 0), borderType=cv2.BORDER_WRAP)

    shift = np.argmax(res)
    print(shift)

    morph_structure = {}
    morph_structure["start"] = start_contour
    morph_structure["end"] = end_contour
    morph_structure["pivot"] = c_end
    morph_structure["shift"] = shift
    morph_structure["offset"] = (c_end[0] - c_start[0], c_end[1] - c_start[1])
    return morph_structure


def to_polar(c, pivot, offset=None):
    if offset is not None:
        x = c[0] + offset[0] - pivot[0]
        y = c[1] + offset[1] - pivot[1]
    else:
        x = c[0] - pivot[0]
        y = c[1] - pivot[1]

    r = math.sqrt(x**2 + y**2)
    t = math.atan2(y, x) + math.pi * 2
    return (r, t)


def from_polar(p, pivot):
    x = p[0] * math.cos(p[1])
    y = p[0] * math.sin(p[1])

    x = x + pivot[0]
    y = y + pivot[1]
    return (x, y)


def unwind(theta):
    if theta > math.pi * 2:
        return theta - math.pi * 2
    return theta


def interpolate_theta(t0, t1, alpha):
    ut0 = unwind(t0)
    ut1 = unwind(t1)
    if abs(t0 - t1) < abs(ut0 - ut1):
        return t0 * (1 - alpha) + t1 * alpha
    else:
        return ut0 * (1 - alpha) + ut1 * alpha


def interpolate(canvas, morph_structure, time):
    start = morph_structure["start"]
    end = morph_structure["end"]
    pivot = morph_structure["pivot"]
    shift = morph_structure["shift"]
    offset = morph_structure["offset"]

    time = 1.0 / (1 + math.exp(-(time - 0.5) * 10))
    canvas.fill(255)
    contourSize = len(start)

    contours = np.zeros((1, contourSize, 2), dtype=np.int32)
    for i in range(contourSize):
        s = to_polar(start[(i + shift + contourSize) % contourSize], pivot, offset)
        e = to_polar(end[(i + 0 + contourSize) % contourSize], pivot)

        t = from_polar((s[0] * (1 - time) + e[0] * time, interpolate_theta(s[1], e[1], time)), pivot)
        contours[0, i, 0] = int(t[0])
        contours[0, i, 1] = int(t[1])

    cv2.fillPoly(canvas, contours, (0, 0, 0, 255))
    return canvas


def extract_contour(image):
    im2, contours, hierarchy = cv2.findContours(255 - image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    largest = None
    c_size = 0
    for i in range(len(contours)):
        if c_size < len(contours[i]):
            c_size = len(contours[i])
            largest = contours[i]
    return largest


def draw_contour(binary_img, contour, wnd_name):
    if contour is None:
        return
    canvas = np.repeat(np.expand_dims(binary_img, 2), 3, 2)
    cv2.drawContours(canvas, [contour], 0, (0, 255, 0), 3)
    cv2.imshow(wnd_name, canvas)


def to_float_mat(raw):
    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    raw = cv2.blur(raw, (7, 7))
    return raw


mask = None


def process_raw(raw, flip=True):
    global mask
    if mask is None:
        mask = np.zeros((raw.shape[0], raw.shape[1]), dtype=np.uint8)
    if flip:
        raw = cv2.flip(raw, 1)
    raw = to_float_mat(raw)
    ret, result = cv2.threshold(raw, 120, 255, cv2.THRESH_BINARY)
    result = np.maximum(result, mask)
    return result


sx = 0
sy = 0


def on_mouse(e, x, y, a, b):
    global sx, sy, mask
    if e == cv2.EVENT_RBUTTONDOWN:
        print(x, y)
        sx = x
        sy = y
    if e == cv2.EVENT_RBUTTONUP:
        if mask is not None:
            mask.fill(0)
            dist = (sx - x)**2 + (sy - y)**2
            xv, yv = np.meshgrid(np.arange(0, mask.shape[1], 1), np.arange(0, mask.shape[0], 1))
            indices = (yv - sy)**2 + (xv - sx)**2 > dist
            mask[indices] = 255


file_dir = os.path.dirname(os.path.realpath(__file__))

source = cv2.VideoCapture(0)
raw_target = cv2.imread(os.path.join(file_dir, "assets", "woman.png"), cv2.IMREAD_UNCHANGED)
raw_target = cv2.blur(raw_target, (7, 7))
morph_target = np.zeros((raw_target.shape[0], raw_target.shape[1]), np.uint8)
idx = np.logical_or(raw_target[:, :, 3] < 1, raw_target[:, :, 0] > 200)
morph_target[idx] = 255
morph_target_contour = extract_contour(morph_target)
draw_contour(morph_target, morph_target_contour, "target")
morph_time = 2.0

destination_writer = cv2.VideoWriter(os.path.join(file_dir, "assets", "gen_vid.mp4"), cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 60, (raw_target.shape[1], raw_target.shape[0]))


if __name__ == '__main__':
    print("Morph")
    cv2.namedWindow("raw")
    cv2.setMouseCallback("raw", on_mouse)

    is_morphing = False
    start_time = time.clock()
    key = None
    morph_structure = None
    canvas = None
    write_file = False
    while key != ord('q'):
        if key == ord('g'):
            write_file = True
            key = ord('m')
        if key == ord('m'):
            ret, frame = source.read()
            start_contour = extract_contour(process_raw(frame))
            if canvas is None:
                canvas = np.zeros((raw_target.shape[0], raw_target.shape[1], 3), dtype=np.uint8)
            if ret and frame is not None and morph_target_contour is not None and start_contour is not None and len(start_contour) > 1:
                is_morphing = True
                start_time = time.clock()
                morph_structure = setup(start_contour, morph_target_contour)

        elapsed = time.clock() - start_time
        if elapsed > morph_time:
            is_morphing = False
            write_file = False

        if is_morphing:
            result = interpolate(canvas, morph_structure, elapsed / morph_time)
            if write_file:
                destination_writer.write(result)
            cv2.imshow("raw", result)
        else:
            ret, frame = source.read()
            processed = process_raw(frame)
            cv2.imshow("raw", processed)
            contour = extract_contour(processed)
            draw_contour(processed, contour, "raw contour")
        key = cv2.waitKey(1)

    source.release()
    destination_writer.release()
