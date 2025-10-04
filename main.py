import os
import argparse
import numpy as np
import random
import sys
from tqdm import tqdm
import torch
import torch.nn as nn

from Configs.builder import get_configs
from Models.builder import get_models
from Datasets.builder import get_datasets
from Opts.builder import get_opts
from Losses.builder import get_losses
from Validations.builder import get_validations

from Utils.basic_utils import AverageMeter
from Utils.utils import set_seed, set_log, gpu, save_ckpt, load_ckpt


# ==========================================================
# 📂 Đường dẫn checkpoint (sẽ lưu 1 file duy nhất)
CHECKPOINT_PATH = "/content/drive/MyDrive/UEM/uem_latest.pt"
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
# ==========================================================


parser = argparse.ArgumentParser(description="Uneven Event Modeling")

parser.add_argument(
    "-d", "--dataset_name", default="act", type=str, metavar="DATASET",
    help="dataset name", choices=["tvr", "act"]
)
parser.add_argument("--gpu", default="1", type=str, help="specify gpu device")
parser.add_argument("--eval", action="store_true")
args = parser.parse_args()


# ==========================================================
def train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer, device):
    """
    Train 1 epoch, auto-save checkpoint duy nhất mỗi 100 batch
    """
    if epoch >= cfg["hard_negative_start_epoch"]:
        criterion.cfg["use_hard_negative"] = True
    else:
        criterion.cfg["use_hard_negative"] = False

    loss_meter = AverageMeter()
    model.train()

    train_bar = tqdm(train_loader, desc=f"epoch {epoch}", total=len(train_loader),
                     unit="batch", dynamic_ncols=True)

    for idx, batch in enumerate(train_bar):
        batch = gpu(batch, device=device)

        optimizer.zero_grad()
        input_list = model(batch)
        loss = criterion(input_list, batch)

        loss.backward()
        optimizer.step()
        loss_meter.update(loss.cpu().item())

        train_bar.set_description(
            f"exp: {cfg['model_name']} epoch:{epoch:2d} iter:{idx:3d} loss:{loss:.4f}"
        )

        # 💾 Lưu duy nhất 1 file, ghi đè mỗi 100 batch
        if idx % 100 == 0:
            checkpoint = {
                "epoch": epoch,
                "batch": idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "config": cfg,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"💾 Saved latest checkpoint: {CHECKPOINT_PATH}")

    return loss_meter.avg


# ==========================================================
def val_one_epoch(epoch, context_dataloader, query_eval_loader,
                  model, val_criterion, cfg, optimizer, best_val, loss_meter, logger):

    val_meter = val_criterion(model, context_dataloader, query_eval_loader)

    if val_meter[4] > best_val[4]:
        es = False
        sc = "New Best Model !!!"
        best_val = val_meter
        save_ckpt(model, optimizer, cfg, os.path.join(cfg["model_root"], "best.ckpt"),
                  epoch, best_val)
    else:
        es = True
        sc = "A Relative Failure Epoch"

    logger.info("=" * 100)
    logger.info(f"Epoch: {epoch:2d}    {sc}")
    logger.info(f"Average Loss: {loss_meter:.4f}")
    logger.info(f"R@1: {val_meter[0]:.1f}, R@5: {val_meter[1]:.1f}, "
                f"R@10: {val_meter[2]:.1f}, Rsum: {val_meter[4]:.1f}")
    logger.info(f"Best Rsum: {best_val[4]:.1f}")
    logger.info("=" * 100)

    return val_meter, best_val, es


# ==========================================================
def validation(context_dataloader, query_eval_loader, model,
               val_criterion, cfg, logger, resume):
    val_meter = val_criterion(model, context_dataloader, query_eval_loader)
    logger.info("=" * 100)
    logger.info(f"Testing from: {resume}")
    logger.info(f"R@1: {val_meter[0]:.1f}, R@5: {val_meter[1]:.1f}, "
                f"R@10: {val_meter[2]:.1f}, Rsum: {val_meter[4]:.1f}")
    logger.info("=" * 100)


# ==========================================================
def main():
    cfg = get_configs(args.dataset_name)
    logger = set_log(cfg["model_root"], "log.txt")
    logger.info(f"Uneven Event Modeling: {cfg['dataset_name']}")

    set_seed(cfg["seed"])
    logger.info(f"set seed: {cfg['seed']}")

    # 🔧 Thiết lập GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"used gpu: {args.gpu}")

    # 📦 Dataset
    logger.info("Loading Data ......")
    cfg, train_loader, context_dataloader, query_eval_loader, \
        test_context_dataloader, test_query_eval_loader = get_datasets(cfg)

    # 🧠 Model
    logger.info("Loading Model ......")
    model = get_models(cfg).to(device)

    optimizer = get_opts(cfg, model, train_loader)
    criterion = get_losses(cfg)
    val_criterion = get_validations(cfg)

    current_epoch, start_batch = -1, 0
    best_val = [0., 0., 0., 0., 0.]
    es_cnt = 0

    # 🔄 Resume nếu có checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        logger.info(f"🔄 Resume from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_epoch = checkpoint["epoch"]
        start_batch = checkpoint["batch"] + 1
    else:
        logger.info("🚀 Starting fresh training...")

    # 🧪 Evaluation mode
    if args.eval:
        if not os.path.exists(CHECKPOINT_PATH):
            logger.info("No trained ckpt found !!!")
        else:
            with torch.no_grad():
                validation(test_context_dataloader, test_query_eval_loader,
                           model, val_criterion, cfg, logger, CHECKPOINT_PATH)
        exit(0)

    # 🔁 Training loop
    for epoch in range(current_epoch + 1, cfg["n_epoch"]):
        loss_meter = train_one_epoch(epoch, train_loader, model, criterion,
                                     cfg, optimizer, device)

        with torch.no_grad():
            val_meter, best_val, es = val_one_epoch(epoch, context_dataloader,
                                                    query_eval_loader, model,
                                                    val_criterion, cfg, optimizer,
                                                    best_val, loss_meter, logger)

        if not es:
            es_cnt = 0
        else:
            es_cnt += 1
            if cfg["max_es_cnt"] != -1 and es_cnt > cfg["max_es_cnt"]:
                logger.info("Early Stop !!!")
                break


# ==========================================================
if __name__ == "__main__":
    main()
