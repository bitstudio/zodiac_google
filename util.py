import numpy as np
import cv2
import math
from PIL import Image
from io import BytesIO
import time
import base64


def convert_1_2_255x3(img):
    return (np.tile(np.reshape(img, (img.shape[0], img.shape[1], 1)) * 255.0, (1, 1, 3))).astype(np.uint8)


def convert_255_2_1(img):
    return img.astype(np.float32) / 255.0


def rgb2gray(rgb):
    return np.expand_dims(np.dot(rgb[..., :3], [0.299, 0.587, 0.114]), 2)


def rgba2gray(rgb):
    return np.expand_dims(np.dot(rgb[..., :3], [0.299, 0.587, 0.114]), 2)


def rgb2bgr(rgb):
    return rgb[..., ::-1]


def bgra2rgb(rgba):
    return (rgba[..., :-1])[..., ::-1]


def all2gray(img):
    if len(img.shape) >= 3:
        if img.shape[2] == 3:
            return rgb2gray(img)
        elif img.shape[2] == 4:
            return rgba2gray(img)
    return np.reshape(img, [img.shape[0], img.shape[1], 1])


def image_to_FC1(img):
    return rgb2gray(convert_255_2_1(img))


def image_to_invFC1(img):
    return 1.0 - rgb2gray(convert_255_2_1(img))


def base64string2array(str_base64):
    return prepareImage(Image.open(BytesIO(base64.b64decode(str_base64))))


def prepareImage(img):
    out = np.asarray(img).astype(np.uint8)
    return out


def image_to_size(img, size):
    img = cv2.resize(img, (size[0], size[1]))
    if len(size) == 2 or size[2] == 1:
        return all2gray(img)
    elif size[2] == 3:
        if len(img.shape) == 2:
            return np.tile(np.expand_dims(img, 2), [1, 1, 3])
        elif img.shape[2] == 1:
            return np.tile(img, [1, 1, 3])
        elif img.shape[2] == 4:
            return img[:, :, :3]
    return img


def threshold(img):
    img[img > 0.5] = 1.0
    img[img <= 0.5] = 0.0
    return img


def to_numpy(lists, size=None, length=None):
    size = (128, 128) if not size else size
    length = len(lists[0]) if not length else length
    m = np.ndarray([len(lists), length, size[0], size[1]])
    for i in range(len(lists)):
        for j in range(length):
            m[i, j, ...] = cv2.resize(lists[i][j], size)
    return m


def resize_batch(arrays, size=None):
    size = (128, 128) if not size else size
    m = np.ndarray([arrays.shape[0], size[0], size[1]])
    for i in range(arrays.shape[0]):
        m[i, ...] = cv2.resize(arrays[i], size)
    return m


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def scan_tokens(s):
    tokens = []
    token = ""
    for i in range(len(s)):
        if s[i] != ' ' and s[i] != '\t' and s[i] != '\n':
            token = token + s[i]
        elif len(token) > 0:
            tokens.append(token)
            token = ""
    return tokens


def load_3d_array(filename):
    with open(filename, 'r') as infile:
        shape = map(int, find_between(infile.readline(), "(", ")").split(","))

        out = np.zeros(shape, dtype=np.float32)
        for b in range(shape[0]):
            for t in range(shape[1]):
                out[b, t, :] = scan_tokens(infile.readline())
            infile.readline()
        return out


def write_3d_array(filename, data):
    with open(filename, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(data.shape))
        for data_slice in data:
            np.savetxt(outfile, data_slice, fmt='%-16.2f')
            outfile.write('# New slice\n')


def make_tile(mat, rows, cols, flip=False):
    b = mat.shape[0]
    r = mat.shape[2] if flip else mat.shape[1]
    c = mat.shape[1] if flip else mat.shape[2]
    shape = (rows, cols, 3) if len(mat.shape) > 3 else (rows, cols)
    canvas = np.zeros(shape, dtype=mat.dtype)
    step = int(max(1, math.floor(b * (r * c) / (rows * cols))))
    i = 0
    for x in range(int(math.floor(rows / r))):
        for y in range(int(math.floor(cols / c))):
            canvas[(x * r):((x + 1) * r), (y * c):((y + 1) * c), ...] = np.transpose(mat[i, ...], (1, 0, 2)) if flip else mat[i, ...]
            i = (i + step) % b

    return canvas


def saveBase64Image(file_name, str_base64):
    with open(file_name + ".png", "wb") as fh:
        fh.write(base64.b64decode(str_base64))


def save_image(img):
    millis = str(int(round(time.time() * 1000)))
    cv2.imwrite("./artifacts/" + millis + ".png", img)
    return millis + ".png"


def np2json(ary):
    strary = "["
    if len(ary.shape) == 1:
        for i in range(ary.shape[0]):
            if i > 0:
                strary = strary + ","
            strary = strary + str(ary[i]).lower()
    else:
        for i in range(ary.shape[0]):
            if i > 0:
                strary = strary + ","
            strary = strary + np2json(ary[i, ...])
    return strary + "]"


if __name__ == '__main__':
    print(np2json(np.asarray([[[1], [2], [3]], [[4], [5], [6]]], dtype=np.float32)))

    print(image_to_size(np.random.rand(2, 2, 1), (4, 4, 1)).shape)
    print(image_to_size(np.random.rand(2, 2, 1), (4, 4, 3)).shape)
    print(image_to_size(np.random.rand(2, 2), (4, 4, 1)).shape)
    print(image_to_size(np.random.rand(2, 2), (4, 4, 3)).shape)
    print(image_to_size(np.random.rand(2, 2, 3), (4, 4, 1)).shape)
    print(image_to_size(np.random.rand(2, 2, 3), (4, 4, 3)).shape)
