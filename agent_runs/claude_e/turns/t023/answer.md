Three findings shape the estimation strategy:
- Finger and lab glucose measure the same thing. At the same hour their mean difference is −0.02 mg/dL, with noise growing with level (SD about 5 at 100 and 12 at 220). An insulin effect on one is an effect on the other.
- Lab draws look random. The gaps between them are geometric (about 20% chance per hour), so lab readings sample glucose without depending on its level.
- Finger-sticks are hourly whenever the last reading was ≥150. Below that they come about every 3 hours at random. So complete-case finger estimates at longer lags are biased toward zero, because the patients whose glucose falls lowest drop out.

Plan: estimate the response at each lag with lab readings as the outcome (no selection), and compare with the finger-based estimates. I'll also estimate how a dose shock changes later doses through the protocol, so I can then remove that feedback and recover the fixed-dose response. If lab and finger agree at short lags, and lab shows larger effects at long lags, that confirms the selection bias.

```python
A = {k: Gtr[k].values for k in ['gf','gl','ins','ne','ma','mc']}
nS = len(stays_tr)
def lp(outkey, k, regkey='ins', policy='gf', on_only=True, extra_fe_hour=False, fill0=False):
    rows=[]
    for h in range(0, 25-k):
        y = A[outkey][:, h+1+k] if True else None   # column index = hour+1 (hours start at -1)
        x = A[regkey][:, h+1]; g0 = A[policy][:, h+1]
        if fill0: y = np.nan_to_num(y, nan=0.0)
        rows.append(np.c_[np.full(nS,h), x, g0, y])
    R = np.concatenate(rows); R = R[~np.isnan(R).any(1)]
    if on_only: R = R[R[:,1]>0]
    R = R[(R[:,2]%10)!=0]
    d = pd.DataFrame(R, columns=['h','x','g0','y'])
    b,se,n = fe_reg(d,'y','x','g0'); return b[0], se[0], n
out=[]
for k in range(1,11):
    out.append((k,)+tuple(np.round(lp('gl',k),3))+tuple(np.round(lp('gf',k),3))+tuple(np.round(lp('ins',k,fill0=True),4)))
print(pd.DataFrame(out, columns=['lag','gl_b','gl_se','gl_n','gf_b','gf_se','gf_n','ins_b','ins_se','ins_n']).to_string())
```