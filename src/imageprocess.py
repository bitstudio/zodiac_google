import cv2


def point_sqrdist(p0, p1):
    return (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2


def point_dist(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def interpolate_point(p0, p1, alpha):
    return (p0[0] * (1 - alpha) + p1[0] * (alpha), p0[1] * (1 - alpha) + p1[1] * (alpha))


class Statistic_extractor:

    def __init__(self, param):
        print("init")

    def contour_resampling(self, contour):
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

    def process(self, img):
