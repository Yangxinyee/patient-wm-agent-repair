The glucose_finger bins run from 78 to 259 mg/dL in 9–19 mg/dL steps, with open-ended bins at each end.

Now I'll price the three candidate shifts on the fwd_2 factual rows. I'll treat each prediction as a Gaussian at the model's predicted mean, with spread proportional to the level, fitted per code from the residuals. I'll score the observed fine bin with and without each shift, and average the change over all 5,220 readings, MAP included.

- If (b), the increment of the full effect, costs under about 0.002 nats, the simple design is viable.
- If only (c), the deviation-only increment, is cheap, the edit must be band-relative.

```python
from scipy.stats import norm as N_
th_k = np.array([-2.2,-2.4,-2.4,-2.4,-1.5,-1.1,-0.9,-0.8,-0.6,-0.5,-0.3,-0.1])
edges = {e['code']:np.array(e['fine_bin_edges']) for e in nm_list if e['attribute']=='value'}
sel_ids = np.unique(f2.stay_id); H0=-3; Hn=30
R = np.zeros((len(sel_ids),Hn-H0)); RD = np.zeros_like(R); ix = {s_:i for i,s_ in enumerate(sel_ids)}
for (s_,h_),r_ in insr_.items():
    if s_ in ix and H0<=h_<Hn:
        g_ = pol.get((s_,h_),np.nan); b_ = 1 if g_<=180 else 2 if g_<=220 else 3 if g_<=260 else 4 if g_<=300 else 6
        R[ix[s_],h_-H0]=r_; RD[ix[s_],h_-H0]=r_-1.056*b_
def lev(A,i,t):
    return sum(th_k[L-1]*A[i,t-L-H0] for L in range(1,13) if t-L-H0>=0)
gg = f2[f2.event.isin(['glucose_finger','glucose_lab'])].copy(); gg['ih']=gg.hour.round().astype(int)
gg = gg.sort_values(['stay_id','event','hour']); gg['tlast']=gg.groupby(['stay_id','event']).ih.shift(1)
res_ = {}
for kind in ['level','incr','dev_incr']:
    tot=0.0
    for ev_,sub_ in gg.groupby('event'):
        m_=sub_.predicted_mean.values; y_=sub_.observed.values; r__=y_-m_
        c_ = np.median(np.abs(r__)/m_)/0.6745; sd_=c_*m_
        e_=edges[ev_]; bi=np.searchsorted(e_,y_,side='right'); lo=np.r_[-np.inf,e_][bi]; hi=np.r_[e_,np.inf][bi]
        A_ = RD if kind=='dev_incr' else R
        sh=[]
        for s_,t_,tl in zip(sub_.stay_id,sub_.ih,sub_.tlast):
            i=ix[s_]; L1=lev(A_,i,t_)
            sh.append(L1 if kind=='level' else (L1-lev(A_,i,int(tl)) if np.isfinite(tl) else L1))
        sh=np.array(sh)
        nll0=-np.log(np.clip(N_.cdf((hi-m_)/sd_)-N_.cdf((lo-m_)/sd_),1e-12,1)); nll1=-np.log(np.clip(N_.cdf((hi-m_-sh)/sd_)-N_.cdf((lo-m_-sh)/sd_),1e-12,1))
        tot+=np.sum(nll1-nll0); res_[(kind,ev_)]=(round(c_,3), round(np.mean(nll1-nll0),4), round(np.std(sh),2))
    print(kind, 'dNLL per all readings %.4f'%(tot/len(f2)))
print(res_)
```