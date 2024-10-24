# from .Master import MASTER
# from .StockMixer import StockMixer
from .LSTM import LSTM
# from .Transformer import Transformer
from .GRU import GRU
# from .GATs import GATs
# from .TCN import TCN

def model_select(name):
    name=name.upper()

    if name in ("LSTM"):
        return LSTM
    elif name in ("GRU"):
        return GRU
    # elif name in ("STOCKMIXER"):
    #     return StockMixer
    # elif name in ("LSTM"):
    #     return LSTM
    # elif name in ("Transformer","TRANSFORMER"):
    #     return Transformer
    # elif name in ("GRU"):
    #     return GRU
    # elif name in ("GAT","GATS"):
    #     return GATs
    # elif name in ("TCN"):
    #     return TCN
    
    else:
        raise NotImplementedError