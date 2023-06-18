import cv2
import numpy as np
from PIL import Image
import copy

def remove_objects(img, lower_size=None, upper_size=None):
    # find all objects
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

    sizes = stats[1:, -1]
    _img = np.zeros((labels.shape))

    # process all objects, label=0 is background, objects are started from 1
    for i in range(1, nlabels):

        # remove small objects
        if (lower_size is not None) and (upper_size is not None):
            if lower_size < sizes[i - 1] and sizes[i - 1] < upper_size:
                _img[labels == i] = 255

        elif (lower_size is not None) and (upper_size is None):
            if lower_size < sizes[i - 1]:
                _img[labels == i] = 255

        elif (lower_size is None) and (upper_size is not None):
            if sizes[i - 1] < upper_size:
                _img[labels == i] = 255

    return _img
 
# 画像のオーバーレイ
def overlayImage(src, overlay, location):
    overlay_height, overlay_width = overlay.shape[:2]

    # 背景をPIL形式に変換
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    pil_src = Image.fromarray(src)
    pil_src = pil_src.convert('RGBA')

    # オーバーレイをPIL形式に変換
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
    pil_overlay = Image.fromarray(overlay)
    pil_overlay = pil_overlay.convert('RGBA')

    # 画像を合成
    pil_tmp = Image.new('RGBA', pil_src.size, (255, 255, 255, 0))
    pil_tmp.paste(pil_overlay, location, pil_overlay)
    result_image = Image.alpha_composite(pil_src, pil_tmp)

    # OpenCV形式に変換
    return cv2.cvtColor(np.asarray(result_image), cv2.COLOR_RGBA2BGRA)

def draw_check(input_image, title):
    img_draw = cv2.resize(input_image, None, fx=0.2, fy=0.2)
    cv2.imshow(title, img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cut_water_crown_1():
    original = cv2.imread("metal_panel.jpg")
    imc = copy.deepcopy(original)
    img = cv2.cvtColor( imc , cv2.COLOR_BGR2GRAY)
    # 適応的ヒストグラム平坦化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2 = clahe.apply(img)

    # 画像の切り抜き
    img2 = img[1000:4092,1850:5104]    
    imc = imc[1000:4092,1850:5104]    

    # 閾値の設定
    threshold = 120 #35 # 90
    
    # 二値化
    #ret, img_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_OTSU) #cv2.THRESH_BINARY)
    ret, img_bin = cv2.threshold(img2, threshold, 255, cv2.THRESH_BINARY)    
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_bin = cv2.dilate(img_bin, kernel_d, iterations=1) # 膨張（Dilation）- dilate()
    img_bin = cv2.erode(img_bin, kernel_e, iterations=4)  # 収縮（Erison）- erode()
    
    # 色チャンネルを追加
    mask = np.stack((img_bin,)*3,-1)
    bit_and = cv2.bitwise_and(imc, mask)
    #draw_check(bit_and, "imc")
    
    # 元サイズ画像に貼り付け
    zero = np.zeros((original.shape[0], original.shape[1]), dtype=np.uint8)
    back_img = np.stack((zero,)*3, -1)
    #print(bit_and.shape)
    h,w = bit_and.shape[:2]
    dx = 1000
    dy = 1850
    output = overlayImage(back_img, bit_and, (dy, dx))
    draw_check(output, "output")

    return output

if __name__ == '__main__':
    cut_water_crown_1()
