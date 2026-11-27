Run 102 shows a reading typically arrives 55 minutes (3,300 s) after the bundle before it, which is the hh:05 drug bundle. The longest gap is 3 hours.

Now I'll compute the view for every reading: out, the normalized slot mean after the time decay, the gap embedding, and z_head applied to the prior mean of z. Then I'll run the value head over a grid of type index × gap scaling, using the register layout that mirrors the embedding's numeric columns. I'll compare with fwd_4's predicted means on glucose_finger. A clearly best combination with small error confirms the structure.

```python
X3=X.astype(np.float64); OUT=X3[:,:192]; HH=X3[:,192:960].reshape(-1,4,192)
code_id={'map_cuff':5,'glucose_finger':6,'map_arterial':7,'glucose_lab':10}
NUM={(e['code'],e['attribute']):e for e in V['numeric']}
# registers: last earlier value of same code
Rv=R[R.event.isin(code_id)].copy(); Rv['min']=np.round(Rv.hour*60).astype(int)
last={}
reg=np.zeros((len(f4),4))  # has, age_s, value, exists
for k,(s,ev,m) in enumerate(zip(f4.stay_id,f4.event,f4['min'])):
    g=Rv[(Rv.stay_id==s)&(Rv.event==ev)&(Rv['min']<m)]
    if len(g): reg[k]=[1,(m-g['min'].iloc[-1])*60.0,g.value.iloc[-1],1]
f4[['has_last','age_s','last_val']]=reg[:,:3]
print('has_last frac',f4.has_last.mean().round(3))
```