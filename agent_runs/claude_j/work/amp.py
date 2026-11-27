
import numpy as np, torch
def amp_stats(lam, stays):
    res=amp_eval(lam, stays); out=[]
    for (s,k0,k1,lgf,lgc) in res:
        rt=RTs[s]; ev=rt['event'].values
        pf=torch.softmax(lgf,-1).mean(1).numpy(); pc=torch.softmax(lgc,-1).mean(1).numpy()
        for i in range(len(rt)):
            c=CEN[ev[i]]; b=int(rt['bfine'].values[i]) if 'bfine' in rt else None
            out.append((s,ev[i],rt['minute'].values[i],float(pf[i][1:]@c),float(pc[i][1:]@c),pf[i],rt['observed'].values[i],rt['nll'].values[i]))
    return out
