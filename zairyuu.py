import cv2
import sys
import pytesseract
from utils.utils import *

def another_try(path):
    info = {}
    # crop and rotate the card
    img = crop_card(path)
    gray_img = resize(img.copy())
    # then binarize it
    img = remover(img)
    # Resize to height = 1650 x width
    img = resize(255 - img)
    showimg(img, 'Input')
    
    rotated = rotate_img(img.copy())
    gray_img = rotate_img(gray_img.copy())
    showimg(rotated)
    
    info['card_number'] = card_number(rotated, gray_img)
    # print(f"card number: {info['card_number']}")

    ## (5) find and draw the upper and lower boundary of each lines
    # [:, rotated.shape[1]//5:int(rotated.shape[1]*2/5)]
    ## this tmp_img is for getting the lines of text
    tmp_img = rotated[:, int(rotated.shape[1]*0.11):int(rotated.shape[1]*0.55)]
    # showimg(tmp_img, "Horizontal projection")
    tmp_img = denoise(tmp_img)
    # showimg(tmp_img, "Horizontal projection - denoise")
    ## horizontal projection
    import time
    start = time.time()
    hist = cv2.reduce(tmp_img, 1, cv2.REDUCE_AVG).reshape(-1)
    
    th = 0
    H, W = tmp_img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y] <= th and hist[y+1] > th]
    lowers = [y for y in range(H-1) if hist[y] > th and hist[y+1] <= th]
    # print(uppers, lowers)
    print(f"cal line: {time.time() - start}")
    
    ## make sure uppers and lowers have the same len
    len_min = min(len(uppers), len(lowers))
    print(f"len: {len_min}")
    uppers = uppers[:len_min]
    lowers = lowers[:len_min]

    lines = [(up, down) for up, down in zip(uppers, lowers) if down - up >= 40]

    # i = 0
    # while i < len(uppers):
    #     if lowers[i] - uppers[i] < 40:
    #         del uppers[i]
    #         del lowers[i]
    #         i -= 1
    #         print(lowers[i], uppers[i])
    #     i+=1

    img = 255 - rotated.copy()
    rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    tmp_lines = []
    for i in range(len(lines)):
        pre_up, pre_down = lines[i-1]
        up, down = lines[i]
        # print(down-up, up-pre_down)
        if up < 150:
            continue
        if abs(up-pre_down) < 30:
            if pre_down-pre_up < 65 or down-up < 65:
                if((pre_up, pre_down) in tmp_lines):
                    tmp_lines.remove((pre_up, pre_down))
                if pre_down-pre_up < down-up and up > 150:
                    tmp_lines.append((up, down))
                elif pre_up > 150:
                    tmp_lines.append((pre_up, pre_down))
            else:
                tmp_lines.append((up, down))
            # if tmp_lines:
            #     del tmp_lines[-1]
            # else:
            #     tmp_lines.append((up, down))
            #     continue
            # if pre_down-pre_up < down-up and pre_down-pre_up < 65:
            #     tmp_lines.append((up, down))
            # elif pre_down-pre_up > down-up and down-up < 65:
            #     tmp_lines.append((pre_up, pre_down))
            # elif pre_down-pre_up < down-up:
            #     tmp_lines.append((up, down))
            # else:
            #     tmp_lines.append((pre_up, pre_down))
        else:
            tmp_lines.append((up, down))
    for up, down in tmp_lines:
        # print(down-up, up-tmp)
        # if down-up < 65 and up-tmp < 30:
        #     continue
        # tmp = down
        # cnt_line += 1
        cv2.line(rotated, (0, up), (rotated.shape[1], up), (255, 0, 0), 4)
        cv2.line(rotated, (0, down), (rotated.shape[1], down), (0, 255, 0), 4)
    # print(len(tmp_lines))
    print(tmp_lines)
    showimg(rotated)
    # return 

    src_img = {}
    h, w = img.shape[:2]
    start_time = time.time()
    for key, value in hard_code.items():
        line = value[-1] - 1
        if line >= len(tmp_lines):
            break
        up, down = tmp_lines[line]
        start = int(value[0]*w)
        end = int(value[1]*w)

        y2 = last_index_img(img[up:down, start:end]) if key not in ['period_of_stay', 'expiration_date'] else end-start
        # Chữa cháy case bị bóng (thường bị ở tên), this is a poor way.
        # In the future, we will base on the độ chói of the image to decide gray or black and white
        # if key in ['name', 'address']:
        sub_img = gray_img[up:down, start:start+y2]
        if key != 'name':
            sub_img = to_bin(sub_img)
        # else:
        #     sub_img = img[up:down, start:start+y2]
        sub_img = padding(sub_img)
        lang = 'eng' if 'name' in key else 'jpn_best'
        result = img2text(sub_img, key, lang)
        info[key] = result
        src_img[key] = sub_img
        if key == 'expiration_date':
            cv2.imwrite('/Users/binhna/Desktop/expiration_date.png', src_img[key])
    print(f"total prediction time: {time.time() - start_time}")
    for key, value in info.items():
        print(f"{key}: {value}")
        if key in src_img.keys():
            # if key == 'expiration_date:':
        #         cv2.imwrite(
        #             '/Users/binhna/Desktop/expiration_date.png', src_img[key])
            showimg(src_img[key], key)

if __name__ == "__main__":
    path = sys.argv[1]
    another_try(path)
