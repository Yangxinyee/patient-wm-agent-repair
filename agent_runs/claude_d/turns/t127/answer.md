The rebuilt bundles match all 5,192 state rows exactly in stay, minute and atom count, so states line up with readings without error. Next I'll write the head reproduction: time flow, then the prior context, `z_head`, `query_norm`, `with_code` with the register, the MLP plus transition logits, and finally the predicted mean. I'll test a few variants of the unknowns (type index, gap scaling, age units) against the factual CSV's predicted means and fine-bin NLL, using the z prior mean first. The reproduction is exact enough if one variant gives near-zero error (RMS well under 1 mg/dL) and its NLL matches the CSV.

```python
from scipy.special import erf
Wd = {n: load_original(n).numpy() if hasattr(load_original(n),'numpy') else np.asarray(load_original(n)) for n in NAMES}
def gelu_(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
def ln_(x,w,b,eps=1e-5):
    mu=x.mean(-1,keepdims=True); v=((x-mu)**2).mean(-1,keepdims=True); return (x-mu)/np.sqrt(v+eps)*w+b
def lin_(x,n,bias=True):
    y = x@Wd[n+'.weight'].T
    return y+Wd[n+'.bias'] if bias and (n+'.bias') in Wd else y
cid = {c['name']:c['id'] for c in V['codes']}
NUMV = {d['code']:d for d in V['numeric'] if d['attribute']=='value'}
CEN = {k:np.array(v) for k,v in V['value_bin_centres'].items()}
# map states rows
S3x = S3['x'].astype(np.float64)
st['row']=np.arange(len(st)); st=st.sort_values(['stay_id','minute']).reset_index(drop=True)
st['prev_row']=st.groupby('stay_id')['row'].shift(1); st['prev_min']=st.groupby('stay_id')['minute'].shift(1)
F4['minute']=np.round(F4['hour']*60).astype(int)
FR = F4.merge(st[['stay_id','minute','prev_row','prev_min']], on=['stay_id','minute'], how='left')
# register: last value of same code strictly before this bundle
FR = FR.sort_values(['stay_id','event','minute']).reset_index(drop=True)
FR['last_val']=FR.groupby(['stay_id','event'])['observed'].shift(1); FR['last_min']=FR.groupby(['stay_id','event'])['minute'].shift(1)
print(FR['prev_row'].isna().sum(), len(FR))
def zsc_(code,x): d=NUMV[code]; return np.clip((x-d['median'])/d['iqr'],-10,10)
def vbin(code,x,side='left'): e=np.array(NUMV[code]['fine_bin_edges']); return 1+np.searchsorted(e,x,side=side)
ok = FR['prev_row'].notna().values
print(ok.sum())
```