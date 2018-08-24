import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import random


def alter(r, t):
    # shift
    shift = random.randrange(0, r.shape[0])
    r = np.roll(r, shift)
    t = np.roll(t, shift)
    return r, t


def point_sqrdist(p0, p1):
    return (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2


def point_dist(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def interpolate_point(p0, p1, alpha):
    return (p0[0] * (1 - alpha) + p1[0] * (alpha), p0[1] * (1 - alpha) + p1[1] * (alpha))


def point_radian(p0, p1):
    return math.atan2(p0[1] - p1[1], p0[0] - p1[0])


def radian_diff(r0, r1):
    delta = r0 - r1
    sign = -1.0 if delta < 0 else 1.0
    abs_delta = abs(delta)
    while abs_delta >= 2 * math.pi:
        abs_delta = abs_delta - 2 * math.pi
    return sign * abs_delta if abs_delta < - (abs_delta - 2 * math.pi) else sign * (abs_delta - 2 * math.pi)


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


def get_polar_stat(contour):

    sx = 0
    sy = 0
    size = max(len(contour), 10)
    for i in range(size):
        sx = sx + contour[i][0]
        sy = sy + contour[i][1]

    centroid = (float(sx) / size, float(sy) / size)

    r = np.zeros((size), dtype=np.float32)
    t = np.zeros((size), dtype=np.float32)

    for i in range(size):
        r[i] = point_dist(contour[i], centroid)
        t[i] = radian_diff(point_radian(contour[(i + 1) % size], centroid), point_radian(contour[i], centroid))

    return r, t


def get_contour_from_polar_stat(r, t, centroid=(0, 0)):

    size = r.shape[0]
    contour = np.zeros((size, 1, 2), dtype=np.int32)
    st = 0
    for i in range(size):
        st = st + t[i]
        contour[i, 0, 0] = math.floor(r[i] * math.cos(st) + centroid[0])
        contour[i, 0, 1] = math.floor(r[i] * math.sin(st) + centroid[1])
    return contour


def extract_contour(image):
    if image.dtype is not np.uint8:
        image = image.astype(np.uint8)
    im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    largest = None
    c_size = 0
    for i in range(len(contours)):
        if c_size < len(contours[i]):
            c_size = len(contours[i])
            largest = contours[i]
    return largest


def blur(raw):
    if raw.shape[2] >= 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    raw = cv2.blur(raw, (7, 7))
    return raw


def process_raw(raw, size, sx, sy, ex, ey, flip=True):
    if ex is None or ey is None:
        ex = raw.shape[1]
        ey = raw.shape[0]
    raw = raw[sy: ey, sx: ex]
    if flip:
        raw = cv2.flip(raw, 1)
    raw = cv2.resize(raw, size)
    raw = blur(raw)
    ret, result = cv2.threshold(raw, 120, 255, cv2.THRESH_BINARY)
    return result


def normalize_distance(r):
    return r / np.mean(r)


def normalize_distance_delta_theta(r, t):
    return r, t * r


def draw_contour(canvas, contour, wnd_name):
    if contour is None:
        return
    cv2.drawContours(canvas, [contour], 0, (0, 255, 0), 3)
    cv2.imshow(wnd_name, canvas)


sx = 0
sy = 0
ex = None
ey = None


def on_mouse(e, x, y, a, b):
    global sx, sy, ex, ey
    if e == cv2.EVENT_LBUTTONDOWN:
        sx = x
        sy = y
    if e == cv2.EVENT_LBUTTONUP:
        ex = x
        ey = y


if __name__ == '__main__':

    print("Generator: 'q' to quit.")
    cv2.namedWindow("raw")
    cv2.setMouseCallback("raw", on_mouse)

    source = cv2.VideoCapture(0)
    key = None
    while key != ord('q'):

        ret, frame = source.read()
        cv2.imshow("raw", frame)
        processed = process_raw(frame, (640, 480), sx, sy, ex, ey)
        contour = extract_contour(255 - processed)

        draw = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        draw_contour(draw, contour, "contour")

        if key == ord('c'):
            eqi_length = re_contour(contour, 256)
            r, t = get_polar_stat(eqi_length)
            r = normalize_distance(r) * 100
            r, t = alter(r, t)
            contour_gen = get_contour_from_polar_stat(r, t, (320, 240))
            draw_contour(np.zeros((480, 640, 3), dtype=np.uint8), contour_gen, "gen")

            plt.plot(range(256), r, t * r * 10)
            plt.show()

        key = cv2.waitKey(1)

    source.release()
