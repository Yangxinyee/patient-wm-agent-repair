The helper is defined, and `fb_(code, x)` returns a reading's fine bin. Next I will run the amplification scan at lam 1, 2 and 4 on the 100 tuning stays. For each lam I will print the factual NLL per reading, overall and for glucose and MAP, plus the teacher-forced insulin x2 shift in glucose_finger over hours 13–24. As a check, the lam 1 NLL should be close to the tool's NLL. The idea is supported if the shift grows well beyond the baseline of about −1.6 while the NLL rises by a few thousandths at most.

```python
def summ(lam, stays):
    o=amp_stats(lam, stays); nl=[];ng=[];nm=[];sh=[];tl=[]
    for (s,e,mi,mf,mc,pf,obs,tn) in o:
        b=fb_(e,obs); v=-np.log(max(pf[b],1e-12)); nl.append(v); tl.append(tn)
        (ng if e.startswith('glucose') else nm).append(v)
        if e=='glucose_finger' and 780<=mi<1500: sh.append(mc-mf)
    return np.mean(nl),np.mean(ng),np.mean(nm),np.mean(sh),len(sh),np.mean(tl)
stays100=list(BUN)
for lam in [1.0,2.0,4.0]:
    r=summ(lam,stays100); print(lam,'nll %.4f glu %.4f map %.4f | tool %.4f | GFshift %.2f n%d'%(r[0],r[1],r[2],r[5],r[3],r[4]))
set_amp(1.0)
```