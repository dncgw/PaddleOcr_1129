import os
import cv2
import glob
from tqdm import tqdm
from paddleocr import PPStructure,draw_structure_result,save_structure_res
from PIL import Image
table_engine = PPStructure(show_log=True)

img_path = r'D:\ocr\TestPic\adk\wrong_p'
save_path = r'D:\ocr\TestPic\adk\wrong_p_resultAnalysis'
save_folder = r'D:\ocr\TestPic\adk\wrong_p_resultAnalysis'
# test
# img_path = 'D://ocr//ocr//check_data//20210906141657.png'

def boardAnalysisTest(input_path,output_path):
    img = cv2.imread(input_path)
    result = table_engine(img)
    save_structure_res(result, save_folder,os.path.basename(input_path).split('.')[0])
    for line in result:
        line.pop('img')
        # print(line)
    font_path = 'D://ocr//TestPic//ppocr_img//fonts//simfang.ttf' # PaddleOCR下提供字体包
    image = Image.open(input_path).convert('RGB')
    im_show = draw_structure_result(image, result,font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save(output_path)

files = os.listdir(img_path)
imgs = glob.glob(img_path + '\\*.jpg')
file_list = os.listdir(img_path)

for file in tqdm(file_list,total=len(imgs)):
    path_input = img_path + '\\' + file
    path_output = save_path + '\\' + file
    boardAnalysisTest(path_input,path_output)