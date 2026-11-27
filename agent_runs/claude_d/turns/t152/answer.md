I now have states for 400 training stays: selS in fwd_3 and selT in fwd_5–7. Next I'll build one pipeline that, for every glucose and MAP reading in these stays, assembles four things:
- the preceding state row and gap;
- the register (last same-code value and its age);
- the exact head input q (z at the prior mean), the full `with_code`, and the reproduced prediction;
- the lagged dose variables from the hourly panel.

As a check, the selS part should reproduce the CSV to about 2 mg/dL, as before. Then I'll redo the residual regressions on all 400 stays.

```python
def load_states(path):
    S_ = np.load(path, allow_pickle=True)
    st_ = pd.DataFrame({'stay_id':S_['stay_id'].astype(int),'minute':np.round(S_['hour'].astype(np.float64)*60).astype(int),
                        'has_ins':S_['has_insulin'],'has_ne':S_['has_norepinephrine']})
    st_['row']=np.arange(len(st_)); return S_['x'].astype(np.float64), st_
parts=[]; Xs=[]; off=0
for k in [3,5,6,7]:
    x_, st_ = load_states(f'/work/fwd_{k}/states_original.npz'); st_['row']+=off; off+=len(x_); Xs.append(x_); parts.append(st_)
XALL = np.concatenate(Xs); STALL = pd.concat(parts).sort_values(['stay_id','minute']).reset_index(drop=True)
STALL['prev_row']=STALL.groupby('stay_id')['row'].shift(1); STALL['prev_min']=STALL.groupby('stay_id')['minute'].shift(1)
stays400 = sorted(STALL.stay_id.unique()); print(len(stays400), len(XALL))
RD = TRAIN[TRAIN.stay_id.isin(stays400) & TRAIN.event.isin(list(NUMV))][['stay_id','hour','event','value']].rename(columns={'value':'observed'}).copy()
RD['minute']=np.round(RD['hour']*60).astype(int)
RD = RD.merge(STALL[['stay_id','minute','prev_row','prev_min']], on=['stay_id','minute'], how='left')
RD = RD.sort_values(['stay_id','event','minute']).reset_index(drop=True)
RD['last_val']=RD.groupby(['stay_id','event'])['observed'].shift(1); RD['last_min']=RD.groupby(['stay_id','event'])['minute'].shift(1)
RD = RD[RD.prev_row.notna()].reset_index(drop=True)
print(len(RD), RD.event.value_counts().to_dict())
```