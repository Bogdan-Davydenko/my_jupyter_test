import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

# Компактная версия:
# Данные


def max_number(img, max_lvl):
    # Получение чб
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # Получение бинарного изображения
    maxl = 0
    maxn = 0
    for lvl in range(1, max_lvl + 1):
        ret, thresh = cv.threshold(gray, lvl, 255, 0)
        thresh = cv.bitwise_not(thresh)
        # Очистка изображение от шумов
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
        # "Уверенный" передний план
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        number, markers = cv.connectedComponents(sure_fg)
        number = number - 1
        if (number > maxn):
            maxn = number
            maxl = lvl
    return maxl, maxn


def count_cells(img_name, max_lvl, save_fig=False):
    img = cv.imread(img_name)
    assert img is not None, "file could not be read, check with os.path.exists()"
    # Предварительная сегментация, определение уровня бинаризации, максимизирующего найденные клетки
    lvl, gbg = max_number(img, max_lvl)

    # Получение чб
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.title("Ч/б, уровень бинаризации " + str(lvl))
    plt.axis('off')
    plt.imshow(cv.cvtColor(gray, cv.COLOR_GRAY2RGB))
    # Получение бинарного изображения
    ret, thresh = cv.threshold(gray, lvl, 255, 0)
    thresh = cv.bitwise_not(thresh)
    # Очистка изображение от шумов
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    # "Уверенный" задний план
    sure_bg = cv.dilate(opening, kernel, iterations=15)
    # "Уверенный" передний план
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    # Серая зона
    unknown = cv.subtract(sure_bg, sure_fg)

    # Определение количества объектов и маркеров
    number, markers = cv.connectedComponents(sure_fg)
    number = number - 1

    plt.subplot(2, 2, 2)
    plt.title("Обнаружено " + str(number) + " клеток")
    plt.axis('off')
    plt.imshow(cv.cvtColor(unknown, cv.COLOR_GRAY2RGB))

    markers = markers + 1
    markers[unknown == 255] = 0

    # Cегментация
    markers = cv.watershed(cv.cvtColor(gray, cv.COLOR_GRAY2RGB), markers)
    segmented_img = np.array(img)
    segmented_img[markers == -1] = [255, 0, 0]

    #print(number)
    plt.subplot(2, 2, 3)
    plt.title("Маркеры")
    plt.axis('off')
    plt.imshow(markers)

    plt.subplot(2, 2, 4)
    plt.title("Границы")
    plt.axis('off')
    plt.imshow(segmented_img)
    if save_fig:
        plt.savefig(img_name.replace('.jpg', '') + '_segmentation_plots.png')
    else:
        plt.show()
    return number


with open('config.txt') as f:
    max_lvl = int(f.readline())


with open('results.txt', 'w') as f:
    for file in os.listdir():
        if file.endswith(".jpg"):
            n = count_cells(file, max_lvl, True)
            f.write(file + " " + str(n) + "\n")
