import paddle
import pandas
import paddleocr
import numpy as np
from pandas.core.frame import DataFrame
import os
import json

import glob
from tqdm import tqdm
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`


img_path = r'D:\ocr\TestPic\lab\src'
save_path = r'D:\ocr\TestPic\lab\result'

def paddleOcrTest(img_path,save_path,filename):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
    result = ocr.ocr(img_path, cls=True)

    t_file = filename[0:filename.rfind('.')]
    result_df = DataFrame(result)
    with open(os.path.join(save_path,t_file+'.csv'),'w',encoding='utf-8')as f:
        result_df.to_csv(f,index=False)
    # for line in result:
    #     print(line)
    # 显示结果
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='D://ocr//TestPic//ppocr_img//fonts//simfang.ttf')
    im_show = Image.fromarray(im_show)
    output_path = save_path + file
    im_show.save(output_path)

files = os.listdir(img_path)
imgs = glob.glob(img_path + '\\*.jpg')
file_list = os.listdir(img_path)

for file in tqdm(file_list,total=len(imgs)):
    path_input = img_path + '\\' + file
    path_output = save_path + '\\'
    paddleOcrTest(path_input,path_output,file)













