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
        testset_loader,
        optimizer,
        criterion,
        max_epochs=100,
        early_stop=100,
        verbose=1,
        log=None,
        save=None,
        savefig=None
):
    best_score=np.inf
    wait=0
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    ic_list = []
    icir_list = []
    ric_list = []
    ricir_list = []

    valid_ic_list = []
    valid_icir_list = []
    valid_ric_list = []
    valid_ricir_list = []

    for epoch in range(max_epochs):
        train_loss=runner.train_one_epoch(model,trainset_loader,optimizer,criterion)
        val_loss = runner.eval_model(model, valset_loader,criterion)
        test_loss = runner.eval_model(model, testset_loader,criterion)
        metrics=test_model(model, runner, testset_loader, log=log)
        valid_metrics=test_model(model, runner, valset_loader, log=log)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)
        ic_list.append(metrics['IC'])
        icir_list.append(metrics['ICIR'])
        ric_list.append(metrics['RIC'])
        ricir_list.append(metrics['RICIR'])

        valid_ic_list.append(valid_metrics['IC'])
        valid_icir_list.append(valid_metrics['ICIR'])
        valid_ric_list.append(valid_metrics['RIC'])
        valid_ricir_list.append(valid_metrics['RICIR'])
        print_log("Epoch %d, train_loss %.6f, valid_loss %.6f , valid_ic %.6f, valid_icir %.6f, valid_ric %.6f, valid_ricir %.6f, test_loss %.6f, ic %.6f, icir %.6f, ric %.6f, ricir %.6f " % (epoch+1, train_loss, val_loss, valid_metrics['IC'],valid_metrics['ICIR'],valid_metrics['RIC'],valid_metrics['RICIR'], test_loss, metrics['IC'],metrics['ICIR'],metrics['RIC'],metrics['RICIR']),log=log)
        model_temp_dct = os.path.join(save, f"{epoch}.pt")
        torch.save(model.state_dict(),model_temp_dct)
        # if train_loss<train_stop_loss_thred:
        #     break
        # if val_loss < best_score:
        #     wait=0
        #     best_score=val_loss
        #     best_state_dict=copy.deepcopy(model.state_dict())
        # else:
        #     wait+=1
        #     if wait>=early_stop:
        #         print_log("Early stopping -----",log=log)
        #         break
    # if savefig:
    #     plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
    #     plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
    #     plt.plot(range(0, epoch + 1), test_loss_list, "-", label="Test Loss")
    #     plt.title("Epoch-Loss")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.legend()
    #     plt.savefig(savefig)
    # model.load_state_dict(best_state_dict)
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

    seed_everything(args.seed)
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


    print_log(model_name,log=log)
    print_log(dataset, log=log)
    print_log("radom seed %d" % (args.seed),log=log)
    print_log("GPU %d" % (args.gpu_num),log=log)
    print_log(cfg, log=log)

        
    # ------------------------------- load dataset ------------------------------- #
    trainset_loader,valset_loader,testset_loader = DataSet(dataset).data_loader_select(model_name)(cfg["batch_size"],cfg["shuffle_train"],cfg["n_jobs"])
    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/{model_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}")
    if not os.path.exists(save):
        os.makedirs(save)
    savefig = os.path.join(log_path, f"{model_name}-{dataset}-{now}.png")


    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    criterion = loss_select(cfg.get("loss", dataset))(**cfg.get("loss_args", {}))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 8e-6)
    )

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
                testset_loader,
                optimizer,
                criterion,
                max_epochs=cfg.get("epochs", 100),
                early_stop=cfg.get("early_stop", 100),
                verbose=1,
                log=log,
                save=save,
                savefig=savefig
                )
    
    
    metrics=test_model(model, runner, testset_loader, log=log)
    print_log("------- Test -----",log=log)
    print_log(metrics,log=log)
    log.close()
