import cv2
import util
import os
import numpy as np
from datetime import datetime
import imageprocess
import random

api_version = "1.0"


def parse_filename(filename, scale=(1.0, 1.0)):

    tokens = filename.split(".")
    label = int(tokens[1])
    qs = tokens[2].split(",")

    x = (float(qs[0]) + float(qs[2])) / 2
    y = (float(qs[1]) + float(qs[3])) / 2
    w = float(qs[2]) - float(qs[0])
    h = float(qs[3]) - float(qs[1])

    return [label, int(x * scale[0]), int(y * scale[1]), int(w * scale[0]), int(h * scale[1])]


def build_filename(number, label, x1, y1, x2, y2):
    return str(number) + "." + str(label) + "." + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + ".png"


# ONLY ACCEPT BLACK BG and WHITE OBJECT
class DataFormat:
    # size = (width, height, depth)
    def __init__(self, depth, size=(512, 512)):
        self.size = size
        self.depth = depth

    def format(self, img):
        img = util.image_to_size(img, self.size)

        raw = imageprocess.blur(img)
        ret, raw = cv2.threshold(raw, 120, 255, cv2.THRESH_BINARY)
        contour = imageprocess.extract_contour(raw)

        eqi_length = imageprocess.re_contour(contour, self.depth)
        r, t = imageprocess.get_polar_stat(eqi_length)
        r = imageprocess.normalize_distance(r)
        r, t = imageprocess.normalize_distance_delta_theta(r, t)
        series = np.stack([r, t])
        return series

    def read_datafile(self, filepath):
        filename = os.path.basename(filepath)
        img = cv2.imread(filepath)
        scale = (self.size[0] / img.shape[1], self.size[1] / img.shape[0])
        label_box = parse_filename(filename, scale)
        series = self.format(img)
        return series, label_box, filename


def save_training_data(setname, frame, label, x1, y1, x2, y2):

    pwd = os.path.dirname(os.path.abspath(__file__))
    datetime_dir = os.path.join(pwd, "data", api_version, setname, datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(datetime_dir):
        os.makedirs(datetime_dir)
        count = 0
    else:
        count = len([name for name in os.listdir(datetime_dir)])
    filename = os.path.join(datetime_dir, build_filename(count, label, x1, y1, x2, y2))
    print("Saving", filename)
    cv2.imwrite(filename, frame)


def parse_date_string(date_string):
    try:
        return datetime.strptime(date_string, "%Y-%m-%d")
    except ValueError:
        return None
    except TypeError:
        return None


def check_inclusion(date, from_date, to_date):
    if to_date is None:
        to_date = datetime.now()
    if from_date is None:
        return date <= to_date
    else:
        return from_date <= date and date <= to_date


def read_data_directory(formatter, from_date, to_date, set_list):

    data = []
    labels = []

    pwd = os.path.dirname(os.path.abspath(__file__))
    api_dir = os.path.join(pwd, "data", api_version)
    if not os.path.exists(api_dir):
        return data, labels

    for setname in os.listdir(api_dir):
        if setname in set_list or len(set_list) is 0:
            set_dir = os.path.join(api_dir, setname)
            for datename in os.listdir(set_dir):
                if check_inclusion(parse_date_string(datename), from_date, to_date):
                    date_dir = os.path.join(set_dir, datename)
                    for filename in os.listdir(date_dir):
                        ts, label, _ = formatter.read_datafile(os.path.join(date_dir, filename))
                        data.append(ts)
                        labels.append(label)

    data = np.asarray(data, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    return data, labels

def write_to_template_directory(frame, index, label, formatter, template_path):
    pwd = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(pwd, template_path)
    filename = build_filename(index, label, 0, 0, 100, 100)
    print("path: ", template_dir, filename)
    if not os.path.exists(template_dir):
        return
    cv2.imwrite(os.path.join(template_dir, filename), frame)



def read_template_directory(formatter, path, with_flip=False):

    data = []
    labels = []

    pwd = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(pwd, path)
    if not os.path.exists(template_dir):
        return data, labels

    for filename in os.listdir(template_dir):
        ts, label, _ = formatter.read_datafile(os.path.join(template_dir, filename))
        label[1] = 1
        data.append(ts)
        labels.append(label.copy())
        if with_flip:
            data.append(np.flip(ts, axis=-1))
            label[1] = -1
            labels.append(label.copy())

    data = np.asarray(data, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    return data, labels


if __name__ == '__main__':
    formatter = DataFormat(256)
    data, labels = read_data_directory(formatter, None, None, [])
    # data = np.flip(data, axis=-1)
    print(data.shape, labels.shape)

    disp_index = random.randrange(0, data.shape[0])
    r = data[disp_index, 0]
    t = data[disp_index, 1]
    print(r, t)
    contour_gen = imageprocess.get_contour_from_polar_stat(r * 100, t / r, (320, 240))
    imageprocess.draw_contour(np.zeros((480, 640, 3), dtype=np.uint8), contour_gen, "gen")
    cv2.waitKey(-1)
