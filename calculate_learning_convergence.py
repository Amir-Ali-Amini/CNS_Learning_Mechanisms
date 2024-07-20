def CLC(W,w_max,w_min,mod=1):
    if mod==1 :return (((W)*(w_max-W)).sum()/w_max**2)/W.shape[0]
    return (((W-w_min)*(w_max-W)).sum()/w_max)/W.shape[0]