import re
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import pytesseract
import unicodedata
import jaconv
from utils.autocorrect import ac

hard_code = {'name': [0.07, 1, 1], 'date_of_birth': [0.11, 0.38, 2], 'gender': [0.38, 0.5, 2], 'nationality': [0.6, 0.81, 2],
             'address': [0.1, 0.71, 3], 'status_of_residence': [0.11, 0.71, 4], 'status': [0.39, 0.71, 5],
             'period_of_stay': [0.22, 0.71, 6], 'type_permission': [0.14, 0.55, 7], 'permitted_date': [0.13, 0.71, 8],
             'expiration_date': [0.17, 0.55, 9]}

# get the last non-zero element
def last_index_img(img):
    img = 255 - img
    # vertical projection
    hist = cv2.reduce(img, 0, cv2.REDUCE_AVG).reshape(-1)
    for i in reversed(range(len(hist)-1)):
        if hist[i] == 0 and hist[i-1] != 0:
            return i+50

def rotate_img(img):
    pts = cv2.findNonZero(img)
    ret = cv2.minAreaRect(pts)
    (cx, cy), (w, h), ang = ret
    if w < h:
        w, h = h, w
        ang += 90

    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return rotated

def img2text(sub_img, key, lang = 'jpn_best'):
    sub_img = 255 - rotate_img(255 - sub_img.copy())
    result = pytesseract.image_to_string(sub_img, lang=lang)
    result = unicodedata.normalize("NFKC", result)
    result = jaconv.normalize(result, 'NFKC')
    if key != 'name':
        result = result.replace(" ", "").replace("*", "")
    if 'date' in key:
        result = regex_date(result)
    elif 'gender' in key:
        result = regex_gender(result)
    elif 'nationality' == key or 'type_permission' == key:
        result = ac(result, key)
    return result

def regex_date(s: str):
    d = re.compile('(\d{4})[^\d]*(\d{2})[^\d]*(\d{2})', re.IGNORECASE)
    result = re.findall(d, s)
    result = result[0] if result else result
    if len(result) == 3:
        out = f"{result[0]} 年 {result[1]} 月 {result[2]} 日"
    else:
        return s
    return out

def regex_idcard(s: str):
    n = re.compile('([A-Z]{2}\d{8}[A-Z]{2})', re.IGNORECASE)
    result = re.findall(n, s)
    # print("\n\n\n\n\n", result)
    if len(result) == 1:
        return result[0]
    return s

def regex_gender(s: str):
    if 'M' in s or '男' in s or 'N' in s:
        return '男'
    elif 'F' in s or 'E' in s or 'L' in s or '女' in s:
        return '女'

def to_bin(path):
    if isinstance(path, str):
        image = cv2.imread(path, 0)
    else:
        image = path.copy()
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return image


def remover(path, low=0, high=255):
    if isinstance(path, str):
        image = cv2.imread(path, 0)
    else:
        image = path.copy()
    # print(image.mean())
    _, image = cv2.threshold(image, image.mean()/1.4, high, cv2.THRESH_BINARY)
    return image


def showimg(img, title=''):
    img = np.array(img)
    plt.imshow(img, cmap='gray')
    plt.title(f"{title}")
    plt.show()


def get_vertical_profile(im):
    v_prof = np.sum((im), axis=1)
    smoothed_prof = smooth(v_prof, 9)
    plt.plot(smoothed_prof)
    plt.show()
    return smoothed_prof


def filter_small_part(img, _rotate):
    # showimg(img)
    x, y, width, height = cv2.boundingRect(img)
    # print(x,y,width,height)
    # print(img.shape)
    img = img[y:y+height, x:x+width]
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img.astype(np.uint8), connectivity=8)
    new_img = np.zeros_like(img)

    # plt.imshow(img)
    # plt.show()

    for i in range(1, ret):
        cc_x = stats[i, cv2.CC_STAT_LEFT]
        cc_y = stats[i, cv2.CC_STAT_TOP]
        cc_width = stats[i, cv2.CC_STAT_WIDTH]
        cc_height = stats[i, cv2.CC_STAT_HEIGHT]

        if cc_width >= 0.2*width or cc_height >= 0.2*height:
            # if cc_width >= 0.1*width or cc_height >= 0.1*height:
            new_img[labels == i] = 1

        if cc_y <= height*0.2:
            print('SHAPE: ', width, height)
            print('CC: ', cc_width, cc_height)
            if cc_width >= 0.3*width and 0.05*height <= cc_height <= 0.2*height:
                _rotate = False
    return (new_img, _rotate)


def padding(sub_img):
    padd = np.ones((50, sub_img.shape[1]))
    tmp_img = np.concatenate((padd*255, sub_img, padd*255), axis=0)
    return tmp_img


def card_number(rotated, gray_img):
    img_number = rotated[:rotated.shape[0]//3, int(rotated.shape[1]*0.75):]
    gray_img = gray_img[:rotated.shape[0]//3, int(rotated.shape[1]*0.75):]
    h, w = img_number.shape[:2]
    # showimg(img_number)

    # denoise some super big cc and super small cc
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_number, connectivity=8)
    for i in range(1, ret):
        if stats[i, cv2.CC_STAT_WIDTH] > 0.7*w or \
            stats[i, cv2.CC_STAT_WIDTH] < 15 and stats[i, cv2.CC_STAT_HEIGHT] < 15 or \
                stats[i, cv2.CC_STAT_TOP] < h*0.05 or stats[i, cv2.CC_STAT_HEIGHT] < 30:
            img_number[labels == i] = 0

    # horizontal projection
    hist = cv2.reduce(img_number, 1, cv2.REDUCE_AVG).reshape(-1)

    th = 0
    H, W = img_number.shape[:2]
    uppers = [y for y in range(H-1) if hist[y] <= th and hist[y+1] > th]
    lowers = [y for y in range(H-1) if hist[y] > th and hist[y+1] <= th]
    # print(uppers, lowers)
    if uppers and lowers:
        uppers = uppers[0]
        lowers = lowers[0]
    else:
        print("Can't extract number ID")
        return ""
    img_number = 255 - img_number[uppers:lowers, :]
    img_number = padding(img_number)
    # showimg(img_number)
    # showimg(gray_img[uppers-20:lowers+20, :])
    result = pytesseract.image_to_string(
        gray_img[uppers-20:lowers+20, :], lang='eng')
    return regex_idcard(result)

import time
def denoise(rotated, debug = False):
    start = time.time()
    # remove noise
    img_ = rotated.copy()
    img_debug = 255-img_
    H, W = img_.shape[:2]
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_, connectivity=8)
    max_cc = max([stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, ret)])
    for i in range(1, ret):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if debug:
            im2 = cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), 1)
            showimg(im2)
        if w < 15 and h < 15 or w/h > 20 or y < 0.05*H or y > 0.95*H:
            img_[labels == i] = 0
    print(f"denoise time: {time.time() - start}")
    return img_

def resize(rotated):
    h, w = rotated.shape[:2]
    min_edge = 0 if h/w < 1 else 1
    factor = 1
    if rotated.shape[min_edge] < 1650:
        factor = 1650/rotated.shape[min_edge]
    rotated = cv2.resize(rotated, None, fx=factor, fy=factor,
                         interpolation=cv2.INTER_CUBIC)
    return rotated

def crop_card(path, debug=False):
    img = remover(path)
    gray_img = cv2.imread(path, 0)
    # find all contours
    contours, _ = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # max contours ==> the card
    c = max(contours, key=cv2.contourArea)
    # get 4 (maybe 4??) peaks
    approx = cv2.approxPolyDP(c, 0.05*cv2.arcLength(c, True), True)

    if debug:
        img_ = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        img_2 = img_.copy()
        # print(len(approx))
        if len(approx) == 4:
            print(approx)
            cv2.drawContours(img_, [approx], 0, (0, 255, 0), 5)
        showimg(img_)

    rect = cv2.minAreaRect(approx)
    # print("rect: {}".format(rect))
    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # print("bounding box: {}".format(box))
    # cv2.drawContours(img_2, [box], 0, (0, 0, 255), 2)
    # showimg(img_2)
    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    # corrdinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(gray_img, M, (width, height))
    if warped.shape[1] < warped.shape[0]:
        warped = ndimage.rotate(warped, 90)

    showimg(warped)
    return warped

# actually this is a temp function to get the id number from the loads of text box
def text_extraction(path):
    if isinstance(path, str):
        small = crop_card(path)
    else:
        small = path.copy()
    small = resize(small)
    # large = cv2.imread(path)
    rgb = small.copy()
    # small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    showimg(grad)

    # _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = to_bin(grad)
    H, W = bw.shape[:2]
    print(H, W)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 4))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    showimg(connected)
    # return
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(
        connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #For opencv 3+ comment the previous line and uncomment the following line
    #_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)
    result = []
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        if x > H and w/h>5 and h > 50 and y < H//4:
            showimg(rgb[y-20:y+h+20, x-20:x+w+20])
            print(w, h)
            result = regex_idcard(pytesseract.image_to_string(rgb[y-20:y+h+20, x-20:x+w+20], lang='eng'))
            print(result)
        # cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        # r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        # if r > 0.45 and w > 8 and h > 8:
        #     cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
    return result
    # showimg(rgb)


# text_extraction('/Users/binhna/Downloads/zairyu/images/zairyu6.jpg')
