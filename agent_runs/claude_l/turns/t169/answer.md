All 8 `last_numeric` columns have similar norms, so all appear to be trained inputs. Type 1 is clearly the best type (4.05 vs ≥5.7 for the others). Before guessing more feature layouts I'll check where the 4 mg/dL error lives: per code, signed versus absolute, and with versus without a previous value in the register. If readings with no earlier value of the same code match closely, the error is in the register features. If those are also off, the problem is upstream, in the view or the z draws.

```python
rows_=[]
for (s,k,c,pm) in tests:
    rf=reg_feats(BUN[s],k,c); m_,_=pred_mean(VV[(s,k)],c,rf,1,'A',None)
    rows_.append((c, rf['has'], m_-pm, BUN[s][k][0]))
rr_=pd.DataFrame(rows_, columns=['code','has','err','hour'])
print(rr_.groupby(['code','has'])['err'].agg(['count','mean',lambda x: np.median(np.abs(x))]).round(3))
print(rr_.groupby(pd.cut(rr_.hour,[-3,0.5,6,12,25]))['err'].agg(['count','mean',lambda x: np.median(np.abs(x))]).round(3))
```