import shutil
import sys
import os

import torch
import gc
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import models
from tool.data_loader import CustomDataset_class, CustomDataset_regress
import argparse
from tool.logger import setup_logger
from torch.utils import data
import torch.nn as nn
from tool.model import Model_test
from tool.utils import resume_checkpoint, fix_seed

fix_seed(523)
git_name = os.popen("git branch --show-current").readlines()[0].rstrip()

# argparse 라이브러리를 통해 테스트 시 필요한 인자들 설정
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="none",
        type=str,
    )

    parser.add_argument("--equ", type=int, default=[1], choices=[1, 2, 3], nargs="+")

    parser.add_argument("--stop_early", type=int, default=30)

    parser.add_argument(
        "--mode",
        default="class",
        choices=["regression", "class"],
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        default=f"checkpoint/{git_name}",
        type=str,
    )

    parser.add_argument(
        "--epoch",
        default=300,
        type=int,
    )

    parser.add_argument(
        "--res",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
    )

    # 새로 추가된 인자
    parser.add_argument(
        "--model",
        default="resnet",  # 기본값을 "resnet"으로 설정
        type=str,
        help="Specify the model type (e.g., resnet, coatnet, etc.)",
    )

    
    args = parser.parse_args()

    return args

# 설정된 모델과 체크포인트를 불러와 테스트 데이터 평가
def main(args):
    args.check_path = os.path.join(args.output_dir, args.mode, args.name)

    if os.path.isdir(os.path.join(args.check_path, "log", "eval")):
        shutil.rmtree(os.path.join(args.check_path, "log", "eval"))
        
    logger = setup_logger(
        args.name,
        os.path.join(args.check_path, "log", "eval"),
        filename=args.name + ".txt",
    )
    logger.info("Command Line: " + " ".join(sys.argv))

    model_num_class = (
        {"dryness": 5, "pigmentation": 6, "pore": 6, "sagging": 7, "wrinkle": 7}
        if args.mode == "class"
        else {
            "pigmentation": 1,
            "moisture": 1,
            "elasticity_R2": 1,
            "wrinkle_Ra": 1,
            "pore": 1,
        }
    )

    # 각 상태별 resNet50 모델이 초기화 되어 있으며, CustomDataset_class 또는 CustomDataset_regress 클래스를 통해 테스트 데이터를 불러옴.
    model_list = {
        key: models.resnet50(weights=models.ResNet50_Weights.DEFAULT, args=args)
        for key, _ in model_num_class.items()
    }

    for key, model in model_list.items(): 
        model.fc = nn.Linear(model.fc.in_features, model_num_class[key], bias = True)
        model_list.update({key: model})

    model_path = os.path.join(
        os.path.join(args.output_dir, args.mode, args.name), "save_model"
    )
    if os.path.isdir(model_path):
        for path in os.listdir(model_path):
            dig_path = os.path.join(model_path, path)
            if os.path.isfile(os.path.join(dig_path, "state_dict.bin")):
                print(f"\033[92mResuming......{dig_path}\033[0m")
                model_list[path] = resume_checkpoint(
                    args,
                    model_list[path],
                    os.path.join(dig_path, "state_dict.bin"),
                    path,
                    False,
                )

    dataset = (
        CustomDataset_class(args, logger, "test")
        if args.mode == "class"
        else CustomDataset_regress(args, logger)
    )
    resnet_model = Model_test(args, logger)

    model_area_dict = (
        {
            "dryness": ["dryness"],
            "pigmentation": ["pigmentation_forehead", "pigmentation_cheek"],
            "pore": ["pore"],
            "sagging": ["sagging"],
            "wrinkle": ["wrinkle_forehead", "wrinkle_glabellus", "wrinkle_perocular"],
        }
        if args.mode == "class"
        else {
            "pigmentation": ["pigmentation"],
            "moisture": ["forehead_moisture", "cheek_moisture", "chin_moisture"],
            "elasticity_R2": [
                "forehead_elasticity_R2",
                "cheek_elasticity_R2",
                "chin_elasticity_R2",
            ],
            "wrinkle_Ra": ["perocular_wrinkle_Ra"],
            "pore": ["cheek_pore"],
        }
    )

    for key in model_list:
        model = model_list[key].cuda()
        for w_key in model_area_dict[key]:
            testset, _ = dataset.load_dataset("test", w_key)
            testset_loader = data.DataLoader(
                dataset=testset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )
            resnet_model.test(model, testset_loader, w_key) # 상태별로 테스트를 수행
            resnet_model.print_test() # 정확도와 상관 계수 등의 평가 지표 출력
        torch.cuda.empty_cache()
        gc.collect()

    resnet_model.save_value()
    resnet_model.plot_Test_losses(key=key)


if __name__ == "__main__":
    args = parse_args()
    main(args)
