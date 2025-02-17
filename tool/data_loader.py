import copy
import errno
import random
from PIL import Image
import natsort
import torch.nn.functional as F
from torchvision import transforms
import os
import numpy as np
import cv2
import json
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset


def mkdir(path):
    if path == "":
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# 분류용 데이터셋을 정의하는 클래스
class CustomDataset_class(Dataset):
    def __init__(self, args, logger, mode):
        self.args = args
        self.logger = logger
        self.load_list(mode)
        self.generate_datasets()

    # 샐플 반환
    def __len__(self):
        return len(self.sub_path)
    #label 반환
    def __getitem__(self, idx):
        idx_list = list(self.sub_path.keys())
        return idx_list[idx], self.sub_path[idx_list[idx]], self.train_num

    # 학습, 검증, 테스트 데이터셋을 구성 (8:1:1)
    def generate_datasets(self):
        self.train_list, self.val_list, self.test_list = (
            defaultdict(lambda: defaultdict()),
            defaultdict(lambda: defaultdict()),
            defaultdict(lambda: defaultdict()),
        )
        for dig in sorted(self.json_dict.keys()):
            class_dict = self.json_dict[dig]
            for grade in sorted(class_dict.keys()):
                grade_dict = class_dict[grade]
                random_list = list(grade_dict.keys())
                random.shuffle(random_list)

                train_len, val_len = int(len(grade_dict) * 0.8), int(len(grade_dict) * 0.1)
                train_idx, val_idx, test_idx = (
                    random_list[:train_len],
                    random_list[train_len : train_len + val_len],
                    random_list[train_len + val_len :],
                )
                grade_dict = dict(grade_dict)

                for dataset_idx, (idx_list, out_list) in enumerate(
                    zip([train_idx, val_idx, test_idx], [self.train_list, self.val_list, self.test_list])
                ):
                    if dataset_idx == 0:
                        in_list = [grade_dict[idx] for idx in idx_list]
                    else:
                        in_list = list()
                        for idx in idx_list:
                            tt_list = list()
                            for each_value in grade_dict[idx]:
                                if each_value.split("_")[-2] in ["F"]:
                                    tt_list.append(each_value)
                            in_list.append(tt_list)
                    out_list[dig][grade] = in_list


    def load_list(self, mode="train"):
        self.mode = mode
        target_list = [
            "pigmentation",
            "moisture",
            "elasticity_R2",
            "wrinkle_Ra",
            "pore",
            "dryness"
        ]
        self.img_path = "dataset/img"
        self.json_path = "dataset/label"

        sub_path_list = [
            item
            for item in natsort.natsorted(os.listdir(self.img_path))
            if not item.startswith(".")
        ]
        self.logger.info(f"Found sub-paths: {sub_path_list}")

        self.json_dict = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
        self.json_dict_train = copy.deepcopy(self.json_dict)

        for equ_name in sub_path_list:
            if equ_name.startswith(".") or int(equ_name) not in self.args.equ:
                continue

            for sub_fold in tqdm(
                natsort.natsorted(os.listdir(os.path.join("dataset/label", equ_name))),
                desc="path loading..",
            ):
                sub_path = os.path.join(equ_name, sub_fold)
                folder_path = os.path.join("dataset/label", sub_path)

                if sub_fold.startswith(".") or not os.path.exists(
                    os.path.join(self.json_path, sub_path)
                ):
                    continue

                for j_name in os.listdir(folder_path):
                    if self.should_skip_image(j_name, equ_name):
                        continue

                    # with open(os.path.join(folder_path, j_name), "r") as f:
                    #     json_meta = json.load(f)
                    #     self.process_json_meta(
                    #         json_meta, j_name, sub_path, target_list, sub_fold
                    #     )
                    json_file_path = os.path.join(folder_path, j_name)
                    try:
                        with open(json_file_path, "r") as f:
                            json_meta = json.load(f)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to decode JSON: {json_file_path}, Error: {e}")
                        return
                    except FileNotFoundError:
                        self.logger.error(f"JSON file not found: {json_file_path}")
                        return
                    self.process_json_meta(json_meta, j_name, sub_path, target_list, sub_fold)


    # JSON 메타데이터 파일을 처리하여 이미지와 label 정보를 불러온다.
    def process_json_meta(
        self, json_meta, j_name, sub_path, target_list, sub_fold
    ):
        if (
            (
                list(json_meta["annotations"].keys())[0] == "acne"
                or json_meta["images"]["bbox"] is None
            )
            and self.args.mode == "class"
        ) or (
            (json_meta["equipment"] is None or json_meta["images"]["bbox"] is None)
            and self.args.mode == "regression"
        ):
            return

        if self.args.mode == "class":
            for dig_n, grade in json_meta["annotations"].items():
                dig, area = dig_n.split("_")[-1], dig_n.split("_")[-2]

                if dig in ["wrinkle", "pigmentation"] and self.mode == "test":
                    dig = f"{dig}_{area}"

                self.json_dict[dig][str(grade)][sub_fold].append(
                    os.path.join(sub_path, j_name.split(".")[0])
                )
        else:
            for dig_n, value in json_meta["equipment"].items():
                matching_dig = [
                    dig_n for target_dig in target_list if target_dig in dig_n
                ]
                if matching_dig:
                    dig = matching_dig[0]
                    self.json_dict[dig][value][sub_fold].append(
                        [
                            os.path.join(sub_path, j_name.split(".")[0]),
                            value,
                        ]
                    )

    # 이미지와 label을 전처리하여 area_list에 저장
    def save_dict(self, transform):
        # ori_img = cv2.imread(os.path.join("dataset/cropped_img", self.i_path + ".jpg"))
        # pil_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        img_path = os.path.join("dataset/cropped_img", self.i_path + ".jpg")
        ori_img = cv2.imread(img_path)

        if ori_img is None:
            self.logger.warning(f"Failed to load image: {img_path}")
            return  # 이미지 로드 실패 시 현재 이미지를 건너뜀

        try:
            pil_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            self.logger.error(f"Error converting image color: {img_path}, Error: {e}")
            return


        s_list = self.i_path.split("/")[-1].split("_")
        desc_area = (
            "Sub_"
            + s_list[0]
            + "_Equ_"
            + s_list[1]
            + "_Angle_"
            + s_list[2]
            + "_Area_"
            + s_list[3]
        )

        if self.args.mode == "class":
            label_data = int(self.grade)
        else:
            norm_v = self.norm_reg(self.value)
            label_data = norm_v

        pil = Image.fromarray(pil_img.astype(np.uint8))
        patch_img = transform(pil)

        self.area_list.append(
            [patch_img, label_data, desc_area]
        )

    def should_skip_image(self, j_name, equ_name):

        # 왼쪽 눈가/볼 ->  좌 15 & 30도
        # 오른쪽 눈가/볼 -> 우 15 & 30도
        # 턱선 -> 위, 아래

        if equ_name == "01":
            return (
                (
                    j_name.split("_")[2] in ["Ft", "Fb"]
                    and j_name.split("_")[3].split(".")[0]
                    in ["08"]
                )
                or (
                    j_name.split("_")[2] in ["R15", "R30"]
                    and j_name.split("_")[3].split(".")[0] in ["04", "06"]
                )
                or (
                    j_name.split("_")[2] in ["L15", "L30"]
                    and j_name.split("_")[3].split(".")[0] in ["03", "05"]
                )
            )
        else:
            return (
                     (
                    j_name.split("_")[2] == "L"
                    and j_name.split("_")[3].split(".")[0]
                    in ["03", "05"]
                )
                or (
                    j_name.split("_")[2] == "R"
                    and j_name.split("_")[3].split(".")[0]
                    in ["04", "06"]
                )
            )

    # 특정 디지트에 맞는 데이터셋을 생성하여 반환.
    def load_dataset(self, mode, dig):
        data_list = (
            self.train_list
            if mode == "train"
            else self.val_list if mode == "val" else self.test_list
        )
        self.area_list = list()
        self.dig = dig

        transform = transforms.Compose(
            [
                transforms.Resize((self.args.res, self.args.res), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        data_list = dict(data_list)
        if self.args.mode == "class":
            

            grade_num = dict()
            grade_num.update(
                {key: len(value) for key, value in data_list[self.dig].items()}
            )

            # for cb loss
            num_grade = [grade_num[num] for num in sorted(grade_num)]

            for self.grade, class_dict in tqdm(
                data_list[self.dig].items(), desc=f"{mode}_class"
            ):
                for self.idx, sub_folder in enumerate(
                    tqdm(sorted(class_dict), desc=f"{self.dig}_{self.grade}")
                ):
                    for self.i_path in sub_folder:
                        self.save_dict(transform)

        else:
            for full_dig in list(data_list.keys()):
                if dig in full_dig:
                    for self.i_path, self.value in tqdm(
                        data_list[full_dig], desc=f"{self.dig}"
                    ):
                        self.save_dict(transform)

        if self.args.mode == "class":
            return self.area_list, num_grade
        else:
            return self.area_list, 0

    def norm_reg(self, value):
        dig_v = self.dig.split("_")[-1]

        if dig_v == "R2":
            return value

        elif dig_v == "moisture":
            return value / 100

        elif dig_v == "Ra":
            return value / 50

        elif dig_v in ["pigmentation", "count"]:
            return value / 350

        elif dig_v == "pore":
            return value / 2600

        else:
            assert 0, "dig_v is not here"


class CustomDataset_regress(CustomDataset_class):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.load_list(args)
        self.generate_datasets()

    def generate_datasets(self):
        self.train_list, self.val_list, self.test_list = dict(), dict(), dict()
        for dig in sorted(self.json_dict):
            train_sub, val_sub, test_sub = list(), list(), list()
            key_index = sorted(self.json_dict[dig])
            i = 0
            for idx in key_index:
                for _, value_list in self.json_dict[dig][idx].items():
                    for value in value_list:
                        if i % 10 == 8:
                            if value[0].split("_")[-2] in ["F"]:
                                val_sub.append(value)
                        elif i % 10 == 9:
                            if value[0].split("_")[-2] in ["F"]:
                                test_sub.append(value)
                        else:
                            train_sub.append(value)
                    i += 1
            self.train_list[dig], self.val_list[dig], self.test_list[dig]  = train_sub, val_sub, test_sub