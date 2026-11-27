The glucose_finger measurement rule is simple. It is always measured if insulin ran in the previous hour, and otherwise measured with probability 0.30 independent of the glucose level. This makes selection fixable. For lags ≥ 2 an insulin innovation can switch later insulin off (by lowering glucose below 150), which removes low-glucose outcomes from the observed sample and biases the effect toward zero. Weighting observed outcomes by 1/P(measured) = 1 or 1/0.30, depending on the previous hour's insulin, removes this bias.

Next I will compute the insulin reduced form for lags 1–12, both unweighted and IPW, with the cross-fitted innovation, then deconvolve with the rate feedback. If the IPW and unweighted results agree, selection is negligible.

```python
def wols(df, y, X, w):
    d=df.dropna(subset=[y]+X); W=d[w].values
    Xm=np.column_stack([np.ones(len(d))]+[d[c].values for c in X]); Y=d[y].values
    XtWX=np.linalg.inv((Xm*W[:,None]).T@Xm); b=XtWX@(Xm*W[:,None]).T@Y; e=Y-Xm@b
    U=pd.DataFrame(Xm*(W*e)[:,None]).groupby(d.stay.values).sum().values; V=XtWX@(U.T@U)@XtWX
    return pd.Series(b,index=['const']+X), pd.Series(np.sqrt(np.diag(V)),index=['const']+X)
ctrlI=['x','gl0','g02','g1','g2','r1','r2']
def irf_ins(IP, maxs=12):
    redu=[];redw=[];sew=[];drr=[]
    for s in range(0,maxs+1):
        d=IP.copy(); tt=d.t.values+s; ok=tt<T; d=d[ok].copy(); tt=tt[ok]
        d['ry']=ins_r[d.stay.values,tt]; bq,_=wols(d.assign(one=1.0),'ry',ctrlI,'one'); drr.append(bq.x)
        if s>0:
            d['y']=gf[d.stay.values,tt]; prev_on = ins_r[d.stay.values,tt-1]>0
            d['w']=np.where(prev_on,1.0,1/0.30); d['one']=1.0
            bu,_=wols(d,'y',ctrlI,'one'); bw,sw=wols(d,'y',ctrlI,'w'); redu.append(bu.x); redw.append(bw.x); sew.append(sw.x)
    drr=np.array(drr); out={}
    for name,red in [('unw',np.array(redu)),('ipw',np.array(redw))]:
        h=np.zeros(maxs+1)
        for s in range(1,maxs+1): h[s]=(red[s-1]-sum(h[s-m]*drr[m] for m in range(1,s)))/drr[0]
        out[name]=(red,h[1:])
    return out, drr, np.array(sew)
outI, drr_I, sew_I = irf_ins(IP)
for k,(red,h) in outI.items(): print(k,'reduced',np.round(red,3)); print(k,'struct ',np.round(h,3),'cum',np.round(np.cumsum(h),2))
print('feedback',np.round(drr_I,3)); print('se ipw',np.round(sew_I,3))
```