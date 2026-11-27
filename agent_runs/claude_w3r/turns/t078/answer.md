Dividing log(dt) by 10 is clearly right for the head too: without it the error jumps to about 10 mg/dL. The age unit hardly matters. My local value head now matches the model to about 1.9 mg/dL against a noise floor of about 1.0 from the 4 z draws. That is close enough to design a head edit that I then verify on the real model.

The key question is whether the information an edit needs is readable at the moment the model predicts g_{s+1}. That information is this hour's insulin rate r_s and its deviation from the sliding-scale band mean m(g_s). Next I compute the value head's 192 hidden activations for every glucose_finger reading in the 100 states stays, and fit grouped-CV ridge probes for r_s, m(g_s), the deviation r_s − m(g_s) and the previous hour's rate.
- An R² well above the 0.24 of the earlier state probe would mean a head-only tilt is feasible.
- A similarly low R² would mean I must first create a cleaner dose channel.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
# band-mean policy m(g_last) from training data
insT2 = tr[tr.event=='insulin'][['stay_id','hour','rate']].copy(); insT2['dh']=np.floor(insT2.hour).astype(int)
gfT = tr[tr.event=='glucose_finger'][['stay_id','hour','value']].copy()
dec = gfT.copy(); dec['dh']=dec.hour.astype(int)
dec = dec.merge(insT2[['stay_id','dh','rate']], on=['stay_id','dh'], how='left').fillna({'rate':0.0})
bands=[-1,150,180,220,260,300,1e9]; dec['band']=pd.cut(dec.value,bands,right=False,labels=False)
mband = dec.groupby('band').rate.mean(); print('band means', mband.round(3).to_dict())
def m_of(g): return float(mband.iloc[int(np.searchsorted(bands[1:],g,side='right'))]) if g is not None else 0.0
# dataset over all glucose_finger readings in selS
fsa = fac[fac.event=='glucose_finger'].reset_index(drop=True)
rows_h=[]; feats_h=[]
for r in fsa.itertuples():
    i0,dt,lv,age = head_inputs(r)
    st_=recs[(recs.stay_id==r.stay_id)]
    ins_prev = st_[(st_.event=='insulin')&(np.isclose(st_.hour, r.hour-1+5/60))]
    ins_prev2 = st_[(st_.event=='insulin')&(np.isclose(st_.hour, r.hour-2+5/60))]
    g_s = st_[(st_.event=='glucose_finger')&(st_.hour<r.hour-1e-9)].value
    rs = float(ins_prev.rate.iloc[0]) if len(ins_prev) else 0.0; rs2=float(ins_prev2.rate.iloc[0]) if len(ins_prev2) else 0.0
    gl = float(g_s.iloc[-1]) if len(g_s) else None
    rows_h.append((r.stay_id, r.hour, rs, rs2, m_of(gl) if gl is not None else 0.0, gl if gl is not None else np.nan, r.observed, r.predicted_mean))
    out_=torch.tensor(S_['x'][i0,:192]); h_=torch.tensor(S_['x'][i0,192:960]).reshape(4,192)
    hf=h_*torch.exp(-dt*torch.exp(fa)); ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10],dtype=torch.float32),'gap','gelu')
    pr=MLP(ctx,'z_prior','gelu'); view=ctx+L_(pr[:32],'z_head'); q=LN(view+Wo['type_emb.weight'][1],'query_norm')
    num=torch.zeros(8); lb=0
    if lv is not None:
        lb=fbin('glucose_finger','value',lv); num[0]=1; num[1]=math.log(1+age/60)/10; num[2]=zsc('glucose_finger','value',lv); num[3]=1
    wc=q+Wo['embed.code.weight'][6]+L_(num,'last_numeric')+Wo['embed.bins.value.weight'][lb]
    feats_h.append(A_(L_(wc,'attr_heads.value_bin.0'),'gelu').numpy())
DH = pd.DataFrame(rows_h, columns=['stay_id','hour','rs','rs2','m','gl','obs','pm']); FH=np.array(feats_h)
print(DH.shape, FH.shape, 'frac with insulin prev hour', round((DH.rs>0).mean(),3))
```