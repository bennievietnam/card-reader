# zairyu
OCR Zairyu Card

## Requirements:
- cv2 for Python

## Troubeshooting:
- Install OpenCV for Python (not for Ubuntu) using:
``` 
pip3 install opencv-python
```
- Make sure your pip is not too old to read newer versions of OpenCV
```
pip3 install --upgrade pip
```

## How to run the test:
```
cd zairyu
python zairyuu.py images/zairyu1.jpg
```

## How it works
- **Step 1**: Apply threshold and then find the largest rectangle box (the card), then return the gray image contains only the card. (OpenCV)
- **Step 2**: Resize all the image to (-1, 1650)
- **Step 3**: Apply threshold on #step1 image, then again using the _minAreaRect_ to rotate that image a little bit.
- **Step 4**: Top left corner is the region contains card ID, B extracts it in the binary image then uses the gray image to ocr.
- **Step 5**: Get the sub binary_card_image from W/5 -> 3W/5 to denoise (delete some small cc, etc) and return the result_image. B applies horizontal projection on the result_image to get loads of lines.
- **Step 6**: Remove those lines which have height smaller than 40 (text lines we need always have the height taller, I am sure).
- **Step 7**: With those lines, B is working on it to make sure how to get only those lines that contain ***valuable information***, which those lines, B only needs to use some ratio to get the filed that B needs 

**(NOTE: B will try to use gray image to ocr instead of threshold image, because when testing in many cases, B see that threshold image lost the information, while the gray image seems like noise but really effective in term of ocr tho)**
