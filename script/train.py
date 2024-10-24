import argparse
import numpy as np
import os
import torch
import datetime
import time
import matplotlib.pyplot as plt
import yaml
import json
import sys
import copy

sys.path.append("..")
from lib.utils import (
    print_log,
    seed_everything,
    set_cpu_num,
)
from models import model_select
from lib.data_preparation import DataSet
from lib.losses import loss_select
from runners import runner_select



def train(model, 
        runner, 
        trainset_loader,
        valset_loader,
        optimizer,
        criterion,
        max_epochs=40,
        early_stop=10,
        train_stop_loss_thred=0.95,
        verbose=1,
        plot=False,
        log=None,
        save=None,
):
    best_score=np.inf
    wait=0
    for epoch in range(max_epochs):
        train_loss=runner.train_one_epoch(model,trainset_loader,optimizer,criterion)
        val_loss = runner.eval_model(model, valset_loader,criterion)
        print_log("Epoch %d, train_loss %.6f, valid_loss %.6f " % (epoch+1, train_loss, val_loss),log=log)
        metrics=test_model(model, runner, testset_loader, log=log)
        print_log("Epoch %d, ic %.6f, icir %.6f, ric %.6f, ricir %.6f " % (epoch+1, metrics['IC'],metrics['ICIR'],metrics['RIC'],metrics['RICIR']),log=log)
        if train_loss<train_stop_loss_thred:
            break
        if val_loss < best_score:
            wait=0
            best_score=val_loss
            best_state_dict=copy.deepcopy(model.state_dict())
        else:
            wait+=1
            if wait>=early_stop:
                print_log("Early stopping -----",log=log)
                break
    model.load_state_dict(best_state_dict)
    return model

def test_model(model,runner, testset_loader,log=None):
    model.eval()
    metrics= runner.predict(model, testset_loader)
    return metrics

if __name__=="__main__":

    # -------------------------- set running environment ------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="CSI300")
    parser.add_argument("-m", "--model", type=str, default="MASTER")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-c", "--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=1)
    args = parser.parse_args()

    seed_everything(0)
    set_cpu_num(args.cpus)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"
    model_name = args.model.upper()
    model_class = model_select(model_name)
    model_name = model_class.__name__

    with open(f"../configs/{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]


    # -------------------------------- load model -------------------------------- #

    model = model_class(**cfg["model_args"]).to(DEVICE)

        
    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/{model_name}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

        
    # ------------------------------- load dataset ------------------------------- #
    print_log(cfg["model_args"],log=log)
    print_log(dataset, log=log)
    trainset_loader,valset_loader,testset_loader = DataSet(dataset).data_loader_select(model_name)(cfg["batch_size"],cfg["shuffle_train"])
    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/{model_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    criterion = loss_select(cfg.get("loss", dataset))(**cfg.get("loss_args", {}))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 8e-6)
    )
    print(cfg.get("lr",8e-6))

        # weight_decay=cfg.get("weight_decay", 0),
        # eps=cfg.get("eps", 1e-8),
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=cfg.get("milestones", []),
    #     gamma=cfg.get("lr_decay_rate", 0.1),
    #     verbose=False,
    # )

    # ----------------------------- set model runner ----------------------------- #

    runner = runner_select(cfg.get("runner", "basic"))(cfg, device=DEVICE, log=log)
    
    # --------------------------- train and test model --------------------------- #
    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    model=train(model, 
                runner, 
                trainset_loader,
                valset_loader,
                optimizer,
                criterion,
                max_epochs=cfg.get("epochs", 100),
                early_stop=cfg.get("early_stop", 10),
                train_stop_loss_thred=cfg.get("train_stop_loss_thred", 0.95),
                verbose=1,
                plot=False,
                log=log,
                save=save,
                )
    
    
    metrics=test_model(model, runner, testset_loader, log=log)
    print_log("------- Test -----",log=log)
    print_log(metrics,log=log)
    log.close()
