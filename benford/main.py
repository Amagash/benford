# import numpy as np
import os
# import glob
import statistics
import cv2
import scipy
from matplotlib import pyplot as plt
# from scipy import signal

import numpy as np
# from sklearn import datasets, linear_model
# from sklearn.linear_model import LinearRegression
# import statsmodels.api as sm
# from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math




def _count_digit(image):
    count_1 = count_2 = count_3 = count_4 = count_5 = count_6 = count_7 = count_8 = count_9 = 0
    dict_of_digits = {"1": count_1, "2": count_2, "3": count_3, "4": count_4, "5": count_5,
                      "6": count_6, "7": count_7, "8": count_8, "9": count_9}
    for line in image:
        for pixel in line:
            pixel2 = str(pixel)
            digit = pixel2[0]
            if digit in dict_of_digits:
                dict_of_digits[digit] += 1
    total = 0
    count_list = []
    for element in dict_of_digits:
        count_list.append(dict_of_digits[element])
        total = total + dict_of_digits[element]
    proportion_list = []
    for count in count_list:
        perc = count / total * 100
        proportion_list.append(perc)
    return proportion_list


def compute_correlation(x, y):
    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2
    return r_squared


def is_fake(image):
    values = _count_digit(image)
    benford = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
    r_squared = compute_correlation(values, benford)
    if r_squared >= 0.99:
        return r_squared
    # elif 0.98 < r_squared < 0.99:
    #     verdict = "Few anomalies detected, might be due to image compression or pixels modification."
    else:
        return r_squared


def plot_digit_hist(image, title):
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    values, benford = is_fake(image)

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, values, width, label='Input image')
    ax.bar(x + width / 2, benford, width, label="Benford's law")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Proportion')
    ax.set_title(title, wrap=True, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.tight_layout()
    # plt.text(-1.3, 34, title, fontsize=10, wrap=True)
    plt.text(5, 20, 'R = {}'.format(round(compute_correlation(values, benford), 5)), wrap=True)
    # plt.text(5, 15, verdict, wrap=True)
    plt.show()


def compute_gradient(image):
    # img_color = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    return mag


def load_images_from_folder(folder):
    # images = []
    result = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            # images.append(img)
            mag = compute_gradient(img)
            result.append(is_fake(mag))
    return result


if __name__ == '__main__':
    path_fake = '../benford/input/VALIDATION/TAMPERED'
    path_real = '../benford/input/VALIDATION/ORIGINAL'

    array_of_r_fake = load_images_from_folder(path_fake)
    array_of_r_real = load_images_from_folder(path_real)

    mean_fake = sum(array_of_r_fake)/len(array_of_r_fake)
    mean_real = sum(array_of_r_real) / len(array_of_r_real)
    #
    stddev_fake = statistics.stdev(array_of_r_fake)
    stddev_real = statistics.stdev(array_of_r_real)
    #
    variance_fake = np.var(array_of_r_fake)
    variance_real = np.var(array_of_r_real)

    # Data for plotting
    result = scipy.stats.ttest_ind(array_of_r_fake, array_of_r_real, equal_var=False)
    print(result)

    # mu = mean_fake
    # variance = math.sqrt(stddev_fake)
    # x = np.linspace(mu - 3 * variance, mu + 3 * variance, 100)
    # plt.plot(x, stats.norm.pdf(x, mu, variance))
    # plt.show()


    # mag = compute_gradient()
    # plot_digit_hist(mag, title="Digit distribution of the gradient magnitude of the image compared to Benford's law")