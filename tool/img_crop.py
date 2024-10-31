import errno
import json
import cv2
import os
from tqdm import tqdm

def mkdir(path):
    # 현재 폴더("")가 입력되면 생성을 건너뜁니다.
    if path == "":
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

folder_path = "dataset/label"   # Dataset 디렉토리 경로

for equ in os.listdir(folder_path):
    equ_path = os.path.join(folder_path, equ)
    
    # 디렉토리인지 확인
    if os.path.isdir(equ_path):
        for sub in tqdm(os.listdir(equ_path)):
            sub_path = os.path.join(equ_path, sub)
            
            # 디렉토리인지 확인
            if os.path.isdir(sub_path):
                print(folder_path, "!@#$", equ, "QWER", sub)
                print("!@#!@#!@#", sub_path)
                
                for anno_path in os.listdir(sub_path):
                    anno_f_path = os.path.join(sub_path, anno_path)
                    
                    # JSON 파일만 처리
                    if os.path.isfile(anno_f_path) and anno_f_path.endswith(".json"):
                        with open(anno_f_path, "r") as f:
                            anno = json.load(f)
                            
                            # 이미지 파일 경로
                            img_path = os.path.join("dataset/img", equ, sub, anno["info"]["filename"])
                            img = cv2.imread(img_path)
                            if img is None:
                                print(f"Image not found: {img_path}")
                                continue
                            
                            # bbox가 없는 경우 건너뜁니다.
                            if anno["images"]["bbox"] is None:
                                continue
                            
                            # bbox 중심을 계산합니다.
                            bbox = list(map(int, anno["images"]["bbox"]))
                            center_bbox = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                            center_bbox = list(map(int, center_bbox))
                            
                            # 저장 폴더 생성
                            save_dir = os.path.join("dataset/cropped_img", equ, sub)
                            mkdir(save_dir)
                            
                            # facepart가 0인 경우 전체 이미지를 사용
                            if anno["images"]["facepart"] == 0:
                                cropped_img = img
                            else:
                                width, height = bbox[3] - bbox[1], bbox[2] - bbox[0]
                                crop_length = int(max(width, height) / 2)
                                
                                cropped_img = img[
                                    max(center_bbox[1] - crop_length, 0): min(center_bbox[1] + crop_length, img.shape[0]),
                                    max(center_bbox[0] - crop_length, 0): min(center_bbox[0] + crop_length, img.shape[1])
                                ]
                            
                            # 이미지를 256x256 크기로 리사이즈하여 저장
                            resized_img = cv2.resize(cropped_img, (256, 256))
                            save_path = os.path.join(
                                save_dir,
                                anno["info"]["filename"][:-4] + f'_{str(anno["images"]["facepart"]).zfill(2)}' + '.jpg'
                            )
                            cv2.imwrite(save_path, resized_img)
