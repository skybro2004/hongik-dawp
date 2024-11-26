import os, shutil, unicodedata, json
import cv2
import numpy as np
from rembg import remove

MARGIN_SIZE = 0
path = '/'.join(os.path.abspath(__file__).split('/')[:-1])

# 변환한 사진 저장할 디렉토리 생성
try:
    os.mkdir("filtered")
except FileExistsError:
    pass

# 디렉토리 설정
to_file_path = os.path.join(path, "filtered")
JSON_DATA_FOLDER_PATH = os.path.join(path, "Training/Training_라벨링데이터")
IMAGE_FOLDER_PATH = os.path.join(path, "Training")


def read_img(category, subcategory, trash_name, file_name):
    # 카테고리가 있는지 확인
    if not os.path.isdir(os.path.join(IMAGE_FOLDER_PATH, f"[T원천]{category}_{subcategory}_{subcategory}")):
        raise # TODO: 에러 메시지

    image_name = ".".join(file_name.split(".")[:-1]) + ".jpg"
    image_path = os.path.join(IMAGE_FOLDER_PATH, f"[T원천]{category}_{subcategory}_{subcategory}", trash_name, image_name)

    # 이미지를 cv2 객체로 불러옴
    image_file = cv2.imread(image_path)
    # 경로에 한글이 있을 시 버그 나는 것 대응하는 코드
    # img_array = np.fromfile(image_path, np.uint8)
    # image_file = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # cv2.imshow('origin', image_file)
    # cv2.waitKey(0)

    return image_file

def read_json(category, subcategory, trash_name, file_name):
    with open(os.path.join(JSON_DATA_FOLDER_PATH, category, subcategory, trash_name, file_name), 'r') as label_file_raw:
        # json 디코드
        label_file_json = json.load(label_file_raw)
    return label_file_json


def image_filter(category, subcategory, trash_name, file_name):
    with open(os.path.join(JSON_DATA_FOLDER_PATH, category, subcategory, trash_name, file_name+".json"), 'r') as label_file_raw:
        # 카테고리가 있는지 확인
        if not os.path.isdir(os.path.join(IMAGE_FOLDER_PATH, f"[T원천]{category}_{subcategory}_{subcategory}")):
            return

        # json 디코드
        label_file_json = json.load(label_file_raw)
        # 객체가 2개 이상이면 return
        if label_file_json["BoundingCount"]!="1":
            return
        
        # 이미지 불러오기
        image_name = ".".join(file_name.split(".")[:-1]) + ".jpg"
        image_path = os.path.join(IMAGE_FOLDER_PATH, f"[T원천]{category}_{subcategory}_{subcategory}", trash_name, image_name)
        # img_array = np.fromfile(image_path, np.uint8)
        # image_file = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # 경로에 한글이 있을 시 버그 나는 것 대응하는 코드
        image_file = cv2.imread(image_path)
        # cv2.imshow('origin', image_file)
        # cv2.waitKey(0)

        bounding = label_file_json["Bounding"][0]
        try:
            image_position = [
                [int(bounding["x1"]), int(bounding["y1"])],
                [int(bounding["x2"]), int(bounding["y2"])]
            ]
        except KeyError:
            # TODO: 사각형이 아닌 Polygon으로 레이블링된 데이터 처리
            return
        image_size = [
            abs(image_position[1][0] - image_position[0][0]),
            abs(image_position[1][1] - image_position[0][1])
        ]
        
        margin_w = int(image_size[0]*MARGIN_SIZE)
        margin_h = int(image_size[1]*MARGIN_SIZE)
        image_position_with_margin = [
            [
                max(0, image_position[0][0] - margin_w),
                max(0, image_position[0][1] - margin_h)
            ],
            [
                image_position[1][0] + margin_w,
                image_position[1][1] + margin_h
            ]
        ]
        
        # 이미지 크롭
        image_cropped = image_file[
            image_position_with_margin[0][1]:image_position_with_margin[1][1],
            image_position_with_margin[0][0]:image_position_with_margin[1][0]
        ]
        # cv2.imshow("cropped", image_cropped)
        # cv2.waitKey(0)

        # target 디렉토리 생성
        try:
            os.makedirs(os.path.join(to_file_path, category, subcategory))
        except FileExistsError:
            pass

        # 크롭한 이미지 저장
        # cv2.imwrite(
        #     os.path.join(to_file_path, category, subcategory, image_name),
        #     image_cropped
        # )
        
        image_bgremoved = remove_bg(image_cropped)
        cv2.imwrite(
            os.path.join(to_file_path, category, subcategory, image_name),
            image_bgremoved
        )
        return


def remove_bg(image):
    image_bgremoved = remove(image)
    # cv2.imshow("test", image_bgremoved)
    # cv2.waitKey(0)
    
    return image_bgremoved


def save_img():
    pass


for category in os.listdir(JSON_DATA_FOLDER_PATH):
    category = unicodedata.normalize('NFC', category) # 한글 풀어쓰기로 인한 오류 방지
    if category==".DS_Store":
        continue

    category_path = os.path.join(JSON_DATA_FOLDER_PATH, category)
    if not os.path.isdir(category_path):
        continue
    print(category, end='\r')

    for subcategory in os.listdir(category_path):
        subcategory = unicodedata.normalize('NFC', subcategory)
        if subcategory==".DS_Store":
            continue
        if not os.path.isdir(os.path.join(IMAGE_FOLDER_PATH, f"[T원천]{category}_{subcategory}_{subcategory}")):
            continue
        
        subcategory_path = os.path.join(category_path, subcategory)
        if not os.path.isdir(subcategory_path):
            continue
        print()
        print(f"└─{subcategory}", end='\r')
        
        total = len(os.listdir(subcategory_path))
        current = 1

        for trash_name in os.listdir(subcategory_path):
            trash_name = unicodedata.normalize('NFC', trash_name)
            if trash_name==".DS_Store":
                continue
            
            trash_path = os.path.join(subcategory_path, trash_name)
            if not os.path.isdir(trash_path):
                continue
            # print(f"  └─{trash_name}")

            for label_name in os.listdir(trash_path):
                label_name = unicodedata.normalize('NFC', label_name)
                if label_name==".DS_Store":
                    continue
                # print(f"    └─{label_name}")
                image_filter(category, subcategory, trash_name, label_name)
            
            # 진행 과정 출력
            current += 1
            percent = min(100, (current / total) * 100)
            filled_length = 40 * current // total
            bar = '█' * filled_length + '-' * (40 - filled_length)
            print(f"└─{subcategory}:\t |{bar}| {percent:.2f}%", end='\r')
        print()