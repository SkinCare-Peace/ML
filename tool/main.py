from cProfile import label
import os
import sys

# Add a abs path for importing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gc
from torch.utils import data
import shutil
import torch.nn as nn
import numpy as np
from torchvision import models # PyTorch에서 제공하는 사전 학습된 모델을 사용
from tensorboardX import SummaryWriter
from utils import  mkdir, resume_checkpoint, fix_seed, CB_loss
from logger import setup_logger
from tool.data_loader import CustomDataset_class, CustomDataset_regress
from model import Model
import argparse 

fix_seed(523)
git_name = os.popen("git branch --show-current").readlines()[0].rstrip()

# argparse 라이브러리를 사용해 하이퍼파라미터와 설정을 커맨드라인 인자로 받는다.
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="none",
        type=str,
    )

    parser.add_argument("--equ", type=int, default=[1], choices=[1, 2, 3], nargs="+") 

    parser.add_argument("--stop_early", type=int, default=50)

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
        default=200,
        type=int,
    )

    parser.add_argument(
        "--res", # 모델에 입력되는 이미지의 해상도
        default=256, # 입력 이미지가 256*256
        type=int,
    )

    parser.add_argument(
        "--gamma", # 손실 함수의 가중치
        # 예를 들어 Focal Loss나 Class Balanced Loss에서 어려운 샘플에 더 큰 가중치를 부여할 때 gamma가 사용한다.
        # gamma가 클수록 어려운 샘플에 더 높은 가중치가 부여 => 모델이 다양한 난이도의 샘플을 다르게 학습하도록 조절.
        default=2,
        type=int,
    )

    parser.add_argument(
        "--load_epoch",
        default=0, # 처음부터 학습 시작 (중간에 중단된 학습을 재개할 때는 마지막에 저장된 epoch로 설정하여 이어서 학습 가능)
        type=int,
    )

    parser.add_argument(
        "--lr",
        default=0.005, # 학습률 0.005
        type=float,
    )

    parser.add_argument(
        "--batch_size",  # 배치 크기 지정. (모델이 한 번의 학습 단계에서 처리할 데이터 샘플 수)
        default=32, # 키우면 학습 속도가 빨라지지만 메모리 사용량 증가
        type=int, # 모델의 성능과 학습 속도에 영향을 미치는 중요한 하이퍼파라미터 중 하나
    )
    
    parser.add_argument(
        "--num_workers",  # 데이터 로더에서 데이터를 불러오는 데 사용할 CPU 스레드 수 지정.
        #default=8, # 스레드 수가 많으면 빠르게 준비 가능. 너무 높으면 시스템 성능에 무리. (CPU 리소스 충분하면 올리면 됨.)
        default=12,
        type=int,
    )

    parser.add_argument("--reset", action="store_true") # --reset 인수가 설정되면 기존의 체크포인트와 로그 파일을 삭제하고 새로운 학습을 시작

    args = parser.parse_args()

    return args # 모든 인수를 파싱하여 반환


def main(args):
    check_path = os.path.join(args.output_dir, args.mode, args.name)
    log_path = os.path.join("tensorboard", git_name, args.mode, args.name)

    mkdir(check_path)
    mkdir(log_path)
    writer = SummaryWriter(log_path)
    logger = setup_logger(args.name + args.mode, os.path.join(check_path, "log", "train"))
    logger.info(args)

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_num_class = (
        {"dryness":5, "pigmentation": 6, "pore": 6, "sagging": 7, "wrinkle": 7}
        if args.mode == "class"
        else {
            "pigmentation": 1,
            "moisture": 1,
            "elasticity_R2": 1,
            "wrinkle_Ra": 1,
            "pore": 1,
        }
    )
    pass_list = list()

    args.best_loss = {item: np.inf for item in model_num_class}
    args.load_epoch = {item: 0 for item in model_num_class}

    # 각 피부 상태 지표별로 개별 ResNet50 모델을 생성하여 이를 딕셔너리로 저장
    # model.fc는 각 상태별로 다르게 설정되며, 최종 분류 출력 크기를 각 상태별로 조정
    model_list = {
        key: models.resnet50(weights=models.ResNet50_Weights.DEFAULT, args=args)
        for key, _ in model_num_class.items()
    }

    '''
    ### 위에꺼는 resNet, 아래로 변경하면 efficient Net b0 
    from torchvision.models import efficientnet_b0  # Import the EfficientNet model

    model_list = {
        key: efficientnet_b0(weights="IMAGENET1K_V1")  # EfficientNet 모델을 초기화
        for key, _ in model_num_class.items()
    }

    '''

    model_path = os.path.join(check_path, "save_model")
        
    for key, model in model_list.items(): 
        model.fc = nn.Linear(model.fc.in_features, model_num_class[key], bias = True)
        model_list.update({key: model})

    ''' 
    ### 여기 for문부터도,  EfficientNet의 classifier[1]은 마지막 완전 연결 레이어이므로, 이를 model_num_class[key]에 맞게 수정
    
    for key, model in model_list.items(): 
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, model_num_class[key], bias=True)
    model_list.update({key: model})

    '''
        

    args.save_img = os.path.join(check_path, "save_img")
    args.pred_path = os.path.join(check_path, "prediction")

    if args.reset:
        print(f"\033[90mReseting......{check_path}\033[0m")
        if os.path.isdir(check_path):
            shutil.rmtree(check_path)
        if os.path.isdir(log_path):
            shutil.rmtree(log_path)

    if os.path.isdir(model_path):
        for path in os.listdir(model_path):
            dig_path = os.path.join(model_path, path)
            if os.path.isfile(os.path.join(dig_path, "state_dict.bin")):
                print(f"\033[92mResuming......{dig_path}\033[0m")
                model_list[path] = resume_checkpoint(
                    args,
                    model_list[path],
                    os.path.join(model_path, f"{path}", "state_dict.bin"),
                    path, 
                )
                if os.path.isdir(os.path.join(dig_path, "done")):
                    print(f"\043[92mPassing......{dig_path}\043[0m")
                    pass_list.append(path)

    mkdir(model_path)
    mkdir(log_path)
    writer = SummaryWriter(log_path)  # tensorboard는 

    logger = setup_logger(
        args.name + args.mode, os.path.join(check_path, "log", "train")
    )
    logger.info(args)
    logger.info("Command Line: " + " ".join(sys.argv))

    dataset = (
        # 학습 및 검증 데이터를 CustomDataset_class 또는 CustomDataset_regress를 통해 불러오고, 학습 로그를 기록할 로거를 설정
        CustomDataset_class(args, logger, "train")
        if args.mode == "class"
        else CustomDataset_regress(args, logger)
    )

    for key in model_list:
        if key in pass_list:
            continue

        model = model_list[key].cpu()
        print("!@#",key)
        trainset, grade_num = dataset.load_dataset("train", key)
        trainset_loader = data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

        valset, _ = dataset.load_dataset("valid", key)
        valset_loader = data.DataLoader(
            dataset=valset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        resnet_model = Model(
            args,
            model,
            trainset_loader,
            valset_loader,
            logger,
            check_path,
            model_num_class,
            writer,
            key,
            grade_num
        )
        resnet_model.model = resnet_model.model.to(device)  # 모델 디바이스 이동
            # 데이터와 레이블 저장
        resnet_model.data = data
        resnet_model.label = label
        # 주어진 epoch 수만큼 학습을 반복
        for epoch in range(args.load_epoch[key], args.epoch):
            resnet_model.update_e(epoch + 1) if args.load_epoch else None

            # 학습 및 검증
            resnet_model.train()
            # 학습 루프가 끝난 후 호출
            resnet_model.valid()
            
            resnet_model.update_e(epoch + 1)
            resnet_model.reset_log()

            if resnet_model.stop_early(): # model.py의 Model 클래스 확인
                break
            
        resnet_model.plot_losses(key=key)

        # 각 epoch가 끝날 때마다 trainset_loader와 valset_loader를 삭제하여 메모리 관리
        # torch.cuda.empty_cache()를 통해 GPU 메모리도 비운다.
        del trainset_loader, valset_loader

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
    resnet_model.plot_losses(key=key)


if __name__ == "__main__":
    args = parse_args()
    main(args)
