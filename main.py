import numpy as np
import cv2
from PIL import Image, ImageDraw


def grayscale_histogram(image):
    count = 0
    resImg = np.zeros((256), dtype=np.float64)  # make a all float 0 array to count
    for i in range(0, image.shape[0]):  # two layered for to scan all pixel
        for j in range(0, image.shape[1]):
            resImg[image[i][j]] = resImg[image[i][j]] + 1  # find the intensity and put it in the right spot in array
            count = count + 1  # check the amount

    new_resImg = np.zeros((256), dtype=np.float64)  # new another all 0 array
    for i in range(256):
        new_resImg[i] = resImg[i] / max(resImg)  # because need to draw, so resize the array ,make them between 0 and 1

    w, h = 256, 256  # picture size

    img = Image.new("RGB", (w, h))  # new an image

    img3 = ImageDraw.Draw(img)  # start to draw line
    for j in range(256):  # total 256 element in array
        img3.line([(j, 256 - new_resImg[j] * 256 * 0.9), (j, 256)], fill="white",
                  width=0)  # draw bottom to top,so need to adjust
    return img


def grayscale_equation(img):
    count = 0
    cdf = np.zeros(256, dtype=np.float64)
    result = np.zeros(256, dtype=np.float64)

    resImg = np.zeros((256), dtype=np.float64)  # make a all float 0 array to count
    for i in range(0, img.shape[0]):  # two layered for to scan all pixel
        for j in range(0, img.shape[1]):
            resImg[img[i][j]] = resImg[img[i][j]] + 1  # find the intensity and put it in the right spot in array
            count = count + 1  # check the amount

    for att in range(256):  # add to calculate cdf
        if (att == 0):
            cdf[0] = resImg[0]
        else:
            cdf[att] = cdf[att - 1] + resImg[att]
            # print(cdf[att])

    for forfun in range(256):
        result[forfun] = round(((cdf[forfun] - min(cdf)) / (max(cdf) - min(cdf))) * 255)  # cdf function
        # print(result[forfun])

    result1 = [round(x) for x in result]  # convert float array to int

    new_resImg = np.zeros((256), dtype=np.float64)  # new another all 0 array
    for i in range(256):
        new_resImg[result1[i]] = resImg[i] / max(
            resImg)  # because need to draw, so resize the array ,make them between 0 and 1
        # change the index to cdf
    w, h = 256, 256  # picture size

    img = Image.new("RGB", (w, h))  # new an image

    img3 = ImageDraw.Draw(img)  # start to draw line
    for j in range(256):  # total 256 element in array
        img3.line([(j, 256 - new_resImg[j] * 0.9 * 256), (j, 256)], fill="white",
                  width=0)  # draw bottom to top,so need to adjust
    return img

    # return result


def equation(img):
    count = 0
    cdf = np.zeros(256, dtype=np.float64)
    result = np.zeros(256, dtype=np.float64)
    new_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    resImg = np.zeros((256), dtype=np.float64)  # make a all float 0 array to count
    for i in range(0, img.shape[0]):  # two layered for to scan all pixel
        for j in range(0, img.shape[1]):
            resImg[img[i][j]] = resImg[img[i][j]] + 1  # find the intensity and put it in the right spot in array
            count = count + 1  # check the amount

    for att in range(256):  # add to calculate cdf
        if (att == 0):
            cdf[0] = resImg[0]
        else:
            cdf[att] = cdf[att - 1] + resImg[att]
            # print(cdf[att])

    for forfun in range(256):
        result[forfun] = round(((cdf[forfun] - min(cdf)) / (max(cdf) - min(cdf))) * 255)  # cdf function
        # print(result[forfun])

    result1 = [round(x) for x in result]  # convert float array to int

    for new1 in range(0, img.shape[0]):  # two for loop to switch the value in original to cdf value
        for new2 in range(0, img.shape[1]):
            new_img[new1, new2] = result1[img[new1, new2]]

    return new_img


def colorR(image):
    count = 0
    resImg = np.zeros((256), dtype=np.float64)  # make a all float 0 array to count
    for i in range(0, image.shape[0]):  # two layered for to scan all pixel
        for j in range(0, image.shape[1]):
            resImg[image[i][j]] = resImg[image[i][j]] + 1  # find the intensity and put it in the right spot in array
            count = count + 1  # check the amount

    new_resImg = np.zeros((256), dtype=np.float64)  # new another all 0 array
    for i in range(256):
        new_resImg[i] = resImg[i] / max(resImg)  # because need to draw, so resize the array ,make them between 0 and 1

    w, h = 256, 256  # picture size

    img = Image.new("RGB", (w, h))  # new an image

    img3 = ImageDraw.Draw(img)  # start to draw line
    for j in range(256):  # total 256 element in array
        img3.line([(j, 256 - new_resImg[j] * 256 * 0.9), (j, 256)], fill="red",
                  width=0)  # draw bottom to top,so need to adjust

    return img


def colorG(image):
    count = 0
    resImg = np.zeros((256), dtype=np.float64)  # make a all float 0 array to count
    for i in range(0, image.shape[0]):  # two layered for to scan all pixel
        for j in range(0, image.shape[1]):
            resImg[image[i][j]] = resImg[image[i][j]] + 1  # find the intensity and put it in the right spot in array
            count = count + 1  # check the amount

    new_resImg = np.zeros((256), dtype=np.float64)  # new another all 0 array
    for i in range(256):
        new_resImg[i] = resImg[i] / max(resImg)  # because need to draw, so resize the array ,make them between 0 and 1

    w, h = 256, 256  # picture size

    img = Image.new("RGB", (w, h))  # new an image

    img3 = ImageDraw.Draw(img)  # start to draw line
    for j in range(256):  # total 256 element in array
        img3.line([(j, 256 - new_resImg[j] * 256 * 0.9), (j, 256)], fill="green",
                  width=0)  # draw bottom to top,so need to adjust

    return img


def colorB(image):
    count = 0
    resImg = np.zeros((256), dtype=np.float64)  # make a all float 0 array to count
    for i in range(0, image.shape[0]):  # two layered for to scan all pixel
        for j in range(0, image.shape[1]):
            resImg[image[i][j]] = resImg[image[i][j]] + 1  # find the intensity and put it in the right spot in array
            count = count + 1  # check the amount

    new_resImg = np.zeros((256), dtype=np.float64)  # new another all 0 array
    for i in range(256):
        new_resImg[i] = resImg[i] / max(resImg)  # because need to draw, so resize the array ,make them between 0 and 1

    w, h = 256, 256  # picture size

    img = Image.new("RGB", (w, h))  # new an image

    img3 = ImageDraw.Draw(img)  # start to draw line
    for j in range(256):  # total 256 element in array
        img3.line([(j, 256 - new_resImg[j] * 256 * 0.9), (j, 256)], fill="blue",
                  width=0)  # draw bottom to top,so need to adjust

    return img


def colorR2(img):
    count = 0
    cdf = np.zeros(256, dtype=np.float64)
    result = np.zeros(256, dtype=np.float64)

    resImg = np.zeros((256), dtype=np.float64)  # make a all float 0 array to count
    for i in range(0, img.shape[0]):  # two layered for to scan all pixel
        for j in range(0, img.shape[1]):
            resImg[img[i][j]] = resImg[img[i][j]] + 1  # find the intensity and put it in the right spot in array
            count = count + 1  # check the amount

    for att in range(256):  # add to calculate cdf
        if (att == 0):
            cdf[0] = resImg[0]
        else:
            cdf[att] = cdf[att - 1] + resImg[att]
            # print(cdf[att])

    for forfun in range(256):
        result[forfun] = round(((cdf[forfun] - min(cdf)) / (max(cdf) - min(cdf))) * 255)  # cdf function
        # print(result[forfun])

    result1 = [round(x) for x in result]  # convert float array to int

    new_resImg = np.zeros((256), dtype=np.float64)  # new another all 0 array
    for i in range(256):
        new_resImg[result1[i]] = resImg[i] / max(
            resImg)  # because need to draw, so resize the array ,make them between 0 and 1
        # change the index to cdf
    w, h = 256, 256  # picture size

    img = Image.new("RGB", (w, h))  # new an image

    img3 = ImageDraw.Draw(img)  # start to draw line
    for j in range(256):  # total 256 element in array
        img3.line([(j, 256 - new_resImg[j] * 0.9 * 256), (j, 256)], fill="red",
                  width=0)  # draw bottom to top,so need to adjust

    return img


def colorG2(img):
    count = 0
    cdf = np.zeros(256, dtype=np.float64)
    result = np.zeros(256, dtype=np.float64)

    resImg = np.zeros((256), dtype=np.float64)  # make a all float 0 array to count
    for i in range(0, img.shape[0]):  # two layered for to scan all pixel
        for j in range(0, img.shape[1]):
            resImg[img[i][j]] = resImg[img[i][j]] + 1  # find the intensity and put it in the right spot in array
            count = count + 1  # check the amount

    for att in range(256):  # add to calculate cdf
        if (att == 0):
            cdf[0] = resImg[0]
        else:
            cdf[att] = cdf[att - 1] + resImg[att]
            # print(cdf[att])

    for forfun in range(256):
        result[forfun] = round(((cdf[forfun] - min(cdf)) / (max(cdf) - min(cdf))) * 255)  # cdf function
        # print(result[forfun])

    result1 = [round(x) for x in result]  # convert float array to int

    new_resImg = np.zeros((256), dtype=np.float64)  # new another all 0 array
    for i in range(256):
        new_resImg[result1[i]] = resImg[i] / max(
            resImg)  # because need to draw, so resize the array ,make them between 0 and 1
        # change the index to cdf
    w, h = 256, 256  # picture size

    img = Image.new("RGB", (w, h))  # new an image

    img3 = ImageDraw.Draw(img)  # start to draw line
    for j in range(256):  # total 256 element in array
        img3.line([(j, 256 - new_resImg[j] * 0.9 * 256), (j, 256)], fill="green",
                  width=0)  # draw bottom to top,so need to adjust

    return img


def colorB2(img):
    count = 0
    cdf = np.zeros(256, dtype=np.float64)
    result = np.zeros(256, dtype=np.float64)

    resImg = np.zeros((256), dtype=np.float64)  # make a all float 0 array to count
    for i in range(0, img.shape[0]):  # two layered for to scan all pixel
        for j in range(0, img.shape[1]):
            resImg[img[i][j]] = resImg[img[i][j]] + 1  # find the intensity and put it in the right spot in array
            count = count + 1  # check the amount

    for att in range(256):  # add to calculate cdf
        if (att == 0):
            cdf[0] = resImg[0]
        else:
            cdf[att] = cdf[att - 1] + resImg[att]
            # print(cdf[att])

    for forfun in range(256):
        result[forfun] = round(((cdf[forfun] - min(cdf)) / (max(cdf) - min(cdf))) * 255)  # cdf function
        # print(result[forfun])

    result1 = [round(x) for x in result]  # convert float array to int

    new_resImg = np.zeros((256), dtype=np.float64)  # new another all 0 array
    for i in range(256):
        new_resImg[result1[i]] = resImg[i] / max(
            resImg)  # because need to draw, so resize the array ,make them between 0 and 1
        # change the index to cdf
    w, h = 256, 256  # picture size

    img = Image.new("RGB", (w, h))  # new an image

    img3 = ImageDraw.Draw(img)  # start to draw line
    for j in range(256):  # total 256 element in array
        img3.line([(j, 256 - new_resImg[j] * 0.9 * 256), (j, 256)], fill="blue",
                  width=0)  # draw bottom to top,so need to adjust

    return img


img1 = cv2.imread('P1.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # change to gray
img2 = cv2.imread('P2.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.imread('P3.jpg')
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

'''
Q1
'''
img = grayscale_histogram(gray1)
img.save('Q1_grayscale_histogram.jpg')
'''
Q2
'''
img = grayscale_equation(gray1)
img.save('Q2_grayscale_histogram_equation.jpg')

img_after_equation = equation(gray1)
cv2.imwrite('gray1.jpg', gray1)
cv2.imwrite('after_equation.jpg', img_after_equation)
'''
Q3
'''
(B2, G2, R2) = cv2.split(img2)
colorB(B2).save('Q3_original_blue_P2.jpg')
colorG(G2).save('Q3_original_green_P2.jpg')
colorR(R2).save('Q3_original_red_P2.jpg')

'''
Q4 color histogram
'''
(B3, G3, R3) = cv2.split(img3)
colorB2(B3).save('Q4_color_blue_P3.jpg')
colorG2(G3).save('Q4_color_green_P3.jpg')
colorR2(R3).save('Q4_color_red_P3.jpg')

'''
Q4 equation histogram
'''

Blue_3 = equation(B3)
Green_3 = equation(G3)
Red_3 = equation(R3)
merge = cv2.merge([Blue_3, Green_3, Red_3])
cv2.imwrite('merge_p3.jpg', merge)
