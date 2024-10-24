import numpy as np
import pandas as pd

def IC(y_true,y_pred):
    df = pd.DataFrame({
        "pred":y_pred,
        "label":y_true
    })
    ic = df['pred'].corr(df['label'])
    return ic

def RIC(y_true,y_pred):
    df = pd.DataFrame({
        "pred":y_pred,
        "label":y_true
    })
    ric = df['pred'].corr(df['label'],method="spearman")
    return ric

def IC_RIC(y_true,y_pred):
    return(
        IC(y_true,y_pred),
        RIC(y_true,y_pred),
    )