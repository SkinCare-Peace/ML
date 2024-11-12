from collections import defaultdict
import errno
import os
import torch
import torch.nn as nn
import copy
from torch.utils import data
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 평가 지표를 누적하고 계산.
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, batch_size=32):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count

    def update_acc(self, val, num=1):
        self.val = val
        self.sum += val
        self.count += num
        self.avg = self.sum / self.count


def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)

    return e_x / torch.sum(e_x, dim=1).unsqueeze(dim=1)


def mkdir(path):
    # if it is the current folder, skip.
    # otherwise the original code will raise FileNotFoundError
    if path == "":
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.epoch / 2.0)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(model, args, epoch, m_dig, best_loss):
    checkpoint_dir = os.path.join(args.output_dir, args.mode, args.name, str(m_dig))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model_state": model_to_save.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
        },
        os.path.join(checkpoint_dir, "temp_file.bin"),
    )

    os.rename(
        os.path.join(checkpoint_dir, "temp_file.bin"),
        os.path.join(checkpoint_dir, "state_dict.bin"),
    )
    return checkpoint_dir


def resume_checkpoint(args, model, path):
    state_dict = torch.load(path)
    best_loss = state_dict["best_loss"]
    epoch = state_dict["epoch"]
    model.load_state_dict(state_dict["model_state"], strict=False)
    del state_dict
    args.load_epoch = epoch

    args.best_loss = best_loss

    return model

#  테스트용 모델 클래스
# 모델 성능을 평가하고 결과를 저장하는 메서드 포함.
class Model_test(object):
    def __init__(self, args, model_list, testset_loader, logger):
        super(Model_test, self).__init__()
        self.args = args
        self.model_list = model_list
        self.logger = logger
        self.test_loader = testset_loader
        self.count = defaultdict(int)

        # 분류에 대한 평균 정확도와 손실 계산
        self.test_class_acc = {
            "sagging": AverageMeter(),
            "wrinkle_forehead": AverageMeter(),
            "wrinkle_glabellus": AverageMeter(),
            "wrinkle_perocular": AverageMeter(),
            "pore": AverageMeter(),
            "pigmentation_forehead": AverageMeter(),
            "pigmentation_cheek": AverageMeter(),
            "dryness": AverageMeter(),
        }

        # 회귀에 대한 평균 정확도와 손실 계산
        self.test_regresion_mae = {
            "moisture": AverageMeter(),
            "wrinkle": AverageMeter(),
            "elasticity": AverageMeter(),
            "pore": AverageMeter(),
            "count": AverageMeter(),
        }

        self.criterion = (
            nn.CrossEntropyLoss() if self.args.mode == "class" else nn.L1Loss()
        )

        self.phase = None
        self.m_dig = 0
        self.model = None
        self.update_c = 0

        self.pred = list()
        self.gt = list()

    def choice(self, m_dig):
        self.model = copy.deepcopy(self.model_list[m_dig])
        self.m_dig = m_dig

    def acc_avg(self, name):
        return round(self.test_class_acc[name].avg * 100, 2)

    def loss_avg(self, name):
        return round(self.test_regresion_mae[name].avg, 4)

    def print_total(self):
        if self.args.mode == "class":
            # self.logger.info(
            #     f"pigmentation_forehead: {self.acc_avg('pigmentation_forehead')}% // pigmentation_cheek: {self.acc_avg('pigmentation_cheek')}% // wrinkle_forehead: {self.acc_avg('wrinkle_forehead')}% // wrinkle_glabellus: {self.acc_avg('wrinkle_glabellus')}% // wrinkle_perocular: {self.acc_avg('wrinkle_perocular')}% // sagging: {self.acc_avg('sagging')}% // pore: {self.acc_avg('pore')}% // dryness: {self.acc_avg('dryness')}%"
            # )
            # self.logger.info(
            #     f"Total Average Acc => {((self.acc_avg('pigmentation_forehead') + self.acc_avg('pigmentation_cheek') + self.acc_avg('wrinkle_forehead') +  self.acc_avg('wrinkle_glabellus') +  self.acc_avg('wrinkle_perocular') + self.acc_avg('sagging') + self.acc_avg('pore') + self.acc_avg('dryness') ) / 8):.2f}%"
            # )
            pass
        else:
            self.logger.info(
                f"count: {self.loss_avg('count')} // moisture: {self.loss_avg('moisture')} // wrinkle: {self.loss_avg('wrinkle')} // elasticity: {self.loss_avg('elasticity')} // pore: {self.loss_avg('pore')}"
            )
            self.logger.info(
                f"Total Average MAE => {((self.loss_avg('count') + self.loss_avg('moisture') + self.loss_avg('wrinkle') +self.loss_avg('elasticity') + self.loss_avg('pore')) / 5):.3f}"
            )

        pred_d = defaultdict(list)
        gt_d = defaultdict(list)

        for g, p in zip(self.pred, self.gt):
            for v in g[1]:
                gt_d[g[0]].append(v)
            for w in p[1]:
                pred_d[p[0]].append(w)

        for k in pred_d.keys():
            (
                macro_precision,
                macro_recall,
                macro_f1,
                _,
            ) = precision_recall_fscore_support(
                pred_d[k], gt_d[k], average="macro", zero_division=1
            )
            acc = accuracy_score(gt_d[k], pred_d[k])
            self.logger.info(
                f"[{k}] Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}, Acc: {acc * 100:.2f}%"
            )

        self.logger.info("============" * 15)

    def match_img(self, vis_img, img):
        col = self.num % self.col
        row = self.num // self.col
        vis_img[row * 256 : (row + 1) * 256, col * 256 : (col + 1) * 256] = img

        return vis_img

    def nan_detect(self, label):
        nan_list = list()
        for batch_idx, batch_data in enumerate(label):
            for value in batch_data:
                if not torch.isfinite(value):
                    nan_list.append(batch_idx)
        return nan_list

    # 각각 회귀와 분류에 대한 손실과 정확도 계산
    def get_test_loss(self, pred, gt):
        self.test_regresion_mae[self.m_dig].update(
            self.criterion(pred[0], gt).item(), batch_size=pred.shape[0]
        )

        self.pred.append([self.m_dig, pred.item()])
        self.gt.append([self.m_dig, gt.item()])

    # 각각 회귀와 분류에 대한 손실과 정확도르 계산
    def get_test_acc(self, pred, gt):
        pred_v = [item.argmax().item() for item in pred]
        gt_v = [item.item() for item in gt]

        self.pred.append([self.m_dig, pred_v])
        self.gt.append([self.m_dig, gt_v])

        score = sum([p == g for (p, g) in zip(pred_v, gt_v)])

        self.test_class_acc[self.m_dig].update_acc(
            score,
            pred.shape[0],
        )

        # 예측 값과 실제값을 파일에 저장하여 나중에 분석할 수 있도록 함.
    def save_value(self):
        path = os.path.join("prediction", self.args.save_path)
        mkdir(path)
        with open(os.path.join(path, f"pred.txt"), "w") as p:
            with open(os.path.join(path, f"gt.txt"), "w") as g:
                for idx in range(len(self.pred)):
                    for p_v, g_v in zip(self.pred[idx][1], self.gt[idx][1]):
                        p.write(f"{self.pred[idx][0]}, {p_v} \n")
                        g.write(f"{self.gt[idx][0]}, {g_v} \n")
        g.close()
        p.close()

    def test(self):
        with torch.no_grad():
            for self.m_dig, datalist in self.test_loader.items():
                self.model = copy.deepcopy(self.model_list[self.m_dig])
                self.model.eval()
                loader_datalist = data.DataLoader(
                    dataset=copy.deepcopy(datalist),
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    shuffle=False,
                )
                del datalist

                for img, label, _, _ in tqdm(loader_datalist, desc=self.m_dig):
                    img, label = img.to(device), label.to(device)
                    pred = self.model.to(device)(img)

                    if self.args.mode == "class":
                        _ = self.get_test_acc(pred, label)
                    else:
                        _ = self.get_test_loss(pred, label)

        self.print_total()
