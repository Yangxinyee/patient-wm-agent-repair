Run 28 settles some embedding details:
- All 8 `last_numeric` inputs and all 6 `embed.numeric` inputs carry weight, so the register has more fields than the 4 documented.
- Categorical and bin tables use index 0 as an all-zero "absent" row.
- `last_status` has a zero row 0 and a large row 1.
- The flow timescales span 600 s to about 2 years, matching the documented range.
- `type_emb` has 15 rows, which suggests type = event-kind × atom-kind (5 × 3).

Next I'll build a CPU version of the heads from the states I already have. I'll compute the view, query, registers and value-head logits for every glucose_finger reading, and compare the predicted means with the factual tool's values. The unknowns I'll scan are the type index, the activation and the layout of the 8 register inputs. A near-perfect correlation confirms the implementation; otherwise I adjust the hypotheses. I'll also update the notebook.

```python
import torch, torch.nn.functional as F
with open('/work/notes.md','a') as fh:
    fh.write("""
## Model-side findings
- Forward2 (factual, original, train sel2 in /work/sel.json): value NLL 1.40394/reading (n=5802).
- Teacher-forced one-step dose sensitivity of model (within-tier OLS on predicted means): insulin lag1 -0.01 (true -2.0); NE lag1 +0.83 (true +3.8). Model ignores dose magnitude.
- Forward3 states (sel2): insulin rate decodable from state R2 0.80 (within-tier dev 0.56); NE rate weakly (R2 0.35, log-ratio 0.05).
- Plan: reimplement heads (then core) on CPU, validate vs factual/states; design edit offline.
""")
Wt = {n: torch.tensor(W[n]) for n in NAMES}
def lin(x, p): 
    y = x @ Wt[p+'.weight'].T
    return y + Wt[p+'.bias'] if (p+'.bias') in Wt else y
def ln(x, p, eps=1e-5): return F.layer_norm(x, (x.shape[-1],), Wt[p+'.weight'], Wt[p+'.bias'], eps)
ACT = {'gelu':F.gelu, 'relu':F.relu, 'silu':F.silu}
def mlp(x, p, act): return lin(ACT[act](lin(x, p+'.0')), p+'.2')
# build reading table with preceding state
st = st.reset_index(drop=True)
st['row'] = np.arange(len(st))
fr = f2.copy()
fr = fr.merge(st[['stay_id','hour','row']].rename(columns={'row':'row_self'}), on=['stay_id','hour'], how='left')
fr['row_prev'] = fr.row_self - 1
assert (st.loc[fr.row_prev,'stay_id'].values == fr.stay_id.values).all()
fr['dt'] = (fr.hour.values - st.loc[fr.row_prev,'hour'].values)*3600.0
# registers: last earlier value of same code in stay
fr = fr.sort_values(['stay_id','event','hour'])
fr['last_val'] = fr.groupby(['stay_id','event']).observed.shift(1)
fr['last_hour'] = fr.groupby(['stay_id','event']).hour.shift(1)
fr = fr.sort_values(['stay_id','hour','event']).reset_index(drop=True)
num = {(d_['code'], d_['attribute']): d_ for d_ in V['numeric']}
code_id = {c['name']: c['id'] for c in V['codes']}
def zval(code, x): d_ = num[(code,'value')]; return np.clip((x-d_['median'])/d_['iqr'], -10, 10)
def vbin(code, x): e = np.array(num[(code,'value')]['fine_bin_edges']); return 1 + (e[None,:] < np.asarray(x)[:,None]).sum(1)
print(fr.head(3).to_string()); print(len(fr))
```