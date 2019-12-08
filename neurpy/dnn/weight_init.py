import torch as th

def default_weight_initialization(m):
    if isinstance(m,th.nn.Conv1d):
        th.nn.init.normal_(m.weight.data)
        if m.bias is not None:
            th.nn.init.normal_(m.bias.data)
    elif isinstance(m,th.nn.Conv2d):
        th.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            th.nn.init.normal_(m.bias.data)
    elif isinstance(m,th.nn.Conv3d):
        th.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            th.nn.init.normal_(m.bias.data)
    elif isinstance(m,th.nn.ConvTranspose1d):
        th.nn.init.normal_(m.weight.data)
        if m.bias is not None:
            th.nn.init.normal_(m.bias.data)
    elif isinstance(m,th.nn.ConvTranspose2d):
        th.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            th.nn.init.normal_(m.bias.data)
    elif isinstance(m,th.nn.ConvTranspose3d):
        th.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            th.nn.init.normal_(m.bias.data)
    elif isinstance(m,th.nn.BatchNorm1d):
        th.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        th.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m,th.nn.BatchNorm2d):
        th.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        th.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m,th.nn.BatchNorm3d):
        th.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        th.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m,th.nn.Linear):
        th.nn.init.xavier_normal_(m.weight.data)
        th.nn.init.normal_(m.bias.data)
    elif isinstance(m,th.nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                th.nn.init.orthogonal_(param.data)
            else:
                th.nn.init.normal_(param.data)
    elif isinstance(m,th.nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                th.nn.init.orthogonal_(param.data)
            else:
                th.nn.init.normal_(param.data)
    elif isinstance(m,th.nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                th.nn.init.orthogonal_(param.data)
            else:
                th.nn.init.normal_(param.data)
    elif isinstance(m,th.nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                th.nn.init.orthogonal_(param.data)
            else:
                th.nn.init.normal_(param.data)

