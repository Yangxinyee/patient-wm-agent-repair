The training set is built: 1,698 items over 100 stays, about 9.5k glucose and 11k MAP counterfactual readings. It is very consistent.

Under every insulin edit, the model supplies only 10–15% of the structural direct effect, and its share shrinks at ×3. For norepinephrine it supplies only 7–9%. So the correction must provide about 85–90% of each direct effect.

Next I convert each counterfactual reading's needed shift into the logit tilt θ that the edit unit must add. I will solve this exactly per reading, by bisection on the 4-draw-averaged, tilted predicted mean. The tilt vector is (glucose_finger centres − 150)/25 for the insulin unit and (map_arterial centres − 75)/10 for the norepinephrine unit.

I will check the θ distributions to choose the gate offset G that keeps GELU linear. Inputs then go into one design matrix for the ridge fit.

```python
VIN=torch.zeros(17,dtype=torch.float64); VIN[1:]=torch.tensor((CEN['glucose_finger']-150)/25)
VNE=torch.zeros(17,dtype=torch.float64); VNE[1:]=torch.tensor((CEN['map_arterial']-75)/10)
def tilt_mean(lg,ev,v,th):
    P=torch.softmax(lg.double()+th[:,None,None]*v,-1).mean(1); cen=torch.tensor(np.stack([CEN[c] for c in ev]))
    return (P[:,1:]*cen).sum(1)/P[:,1:].sum(1)
def solve_theta(lg,ev,v,T):
    base=tilt_mean(lg,ev,v,torch.zeros(len(T),dtype=torch.float64)); tgt=base+torch.tensor(T)
    lo=torch.full_like(base,-8.); hi=torch.full_like(base,8.)
    for _ in range(45):
        mid=(lo+hi)/2; mm=tilt_mean(lg,ev,v,mid); up=mm<tgt; lo=torch.where(up,mid,lo); hi=torch.where(up,hi,mid)
    return ((lo+hi)/2).numpy()
for it in GEN:
    if it['drug']=='fact': continue
    T=it['st']-(pmean(it['lg'],it['ev'])-pmean(it['lgf'],it['ev']))
    it['T']=T; it['th']=solve_theta(it['lg'],it['ev'],VIN if it['drug']=='insulin' else VNE,T)
for d_ in ['insulin','norepinephrine']:
    th=np.concatenate([it['th'] for it in GEN if it['drug']==d_]); T=np.concatenate([it['T'] for it in GEN if it['drug']==d_])
    print(d_,'theta pct',np.percentile(th,[0.5,5,50,95,99.5]).round(2),'T pct',np.percentile(T,[0.5,50,99.5]).round(1),'frac |th|>7.9',round(float((np.abs(th)>7.9).mean()),4))
```