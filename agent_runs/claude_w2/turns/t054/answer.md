The inspection gave the details I needed:
- **Codes:** map_cuff 5, glucose_finger 6, map_arterial 7, glucose_lab 10.
- **Bin centres:** glucose_finger runs 72 to 271 mg/dL, so a predicted mean is capped at 271.
- **Bundle rows** are in time order, including the admission rows at hours −2 and 0.

Now I fit my re-implementation of the value head to the tool's teacher-forced predicted means. For each reading:
1. Take the state after the preceding bundle and apply the time-flow decay.
2. Build the view, using the z prior mean in place of the 4 random draws.
3. Apply query_norm with a candidate type embedding.
4. Add the code embedding, the last-value register terms and the transition-table row.
5. Run the head and take the mean over bin centres.

Several head inputs are undocumented, so I search over them:
- the type index (if types are event kind × atom kind, a measurement would be type 3);
- the gap input scaling;
- the age unit;
- the layout of the 8 `last_numeric` inputs;
- whether `last_status` is added.

**Success:** a correlation near 1 and errors of about 1 mg/dL for the best variant.

```python
import itertools, time
t0_=time.time()
Wd = {k_: load(k_).double() for k_ in NAMES}
def L(x, pre): return x @ Wd[pre+'.weight'].T + (Wd[pre+'.bias'] if pre+'.bias' in Wd else 0)
def M(x, pre): return L(F.silu(L(x, pre+'.0')), pre+'.2')
def LN_(x, pre): return F.layer_norm(x, (192,), Wd[pre+'.weight'], Wd[pre+'.bias'], 1e-6)
import torch.nn.functional as F
X_ = torch.tensor(S['x'], dtype=torch.float64); OUT=X_[:,:192]; HH=X_[:,192:960].reshape(-1,4,192)
hS = S['hour']; sS = S['stay_id']
key = {(int(s_), round(float(h_),4)): i_ for i_, (s_, h_) in enumerate(zip(sS, hS))}
fa4 = pd.read_csv('/work/fwd_4/factual_original.csv')
fa4['row'] = [key[(int(s_), round(float(h_),4))] for s_, h_ in zip(fa4.stay_id, fa4.hour)]
fa4['prev'] = fa4.row - 1
assert (sS[fa4.prev.values] == fa4.stay_id.values).all()
fa4['dt'] = (fa4.hour.values - hS[fa4.prev.values]) * 3600.0
fa4['code'] = fa4.event.map(code_id)
fa4 = fa4.sort_values(['stay_id','event','hour']).reset_index(drop=True)
fa4['last'] = fa4.groupby(['stay_id','event']).observed.shift(1); fa4['last_h'] = fa4.groupby(['stay_id','event']).hour.shift(1)
num_ = {(d_['code'], d_['attribute']): d_ for d_ in VOCAB['numeric']}
def zs(code, x): d_ = num_[(code,'value')]; return np.clip((x - d_['median'])/d_['iqr'], -10, 10)
def fb(code, x): return 1 + np.searchsorted(np.array(num_[(code,'value')]['fine_bin_edges']), x, side='right')
fa4['has'] = fa4['last'].notna().astype(float)
fa4['zl'] = [zs(e_, v_) if not np.isnan(v_) else 0.0 for e_, v_ in zip(fa4.event, fa4['last'])]
fa4['lb'] = [fb(e_, v_) if not np.isnan(v_) else 0 for e_, v_ in zip(fa4.event, fa4['last'])]
fa4['age_s'] = ((fa4.hour - fa4.last_h)*3600).fillna(0)
cen = {c_: np.array(v_) for c_, v_ in VOCAB['value_bin_centres'].items()}
P_ = torch.tensor(fa4.prev.values); dt_ = torch.tensor(fa4.dt.values)
dec = torch.exp(-dt_[:,None] * torch.exp(Wd['core.flow.a'])[None,:])
hfl = HH[P_] * dec[:,None,:]
base0 = OUT[P_] + LN_(hfl.mean(1), 'core.out_norm')
def gapv(mode):
    x_ = {'log': torch.log(dt_), 'log10': torch.log(dt_)/10, 'log1p': torch.log1p(dt_)}[mode]
    return M(x_[:,None], 'gap')
codes_t = torch.tensor(fa4.code.values); lb_t = torch.tensor(fa4.lb.values)
cenmat = np.stack([cen[e_] for e_ in fa4.event]); tgt = fa4.predicted_mean.values
def run(gmode='log', ty=3, status=None, layout='A', age_unit=1.0, zmode='mean'):
    ctx_ = base0 + gapv(gmode)
    zp = M(ctx_, 'z_prior'); zmu = zp[:, :32]
    view = ctx_ + L(zmu, 'z_head') if zmode=='mean' else ctx_
    q_ = LN_(view + Wd['type_emb.weight'][ty], 'query_norm')
    la = np.log1p(fa4.age_s.values/age_unit)/10 * fa4.has.values
    f = np.zeros((len(fa4), 8))
    if layout=='A': f[:,0]=fa4.has; f[:,1]=la; f[:,2]=fa4.zl
    if layout=='B': f[:,0]=fa4.has; f[:,1]=la; f[:,2]=fa4.zl; f[:,4]=fa4.has
    if layout=='C': f[:,0]=fa4.has; f[:,1]=la; f[:,2]=fa4.zl; f[:,3]=fa4.has
    wc = q_ + Wd['embed.code.weight'][codes_t] + L(torch.tensor(f), 'last_numeric') + Wd['embed.bins.value.weight'][lb_t] + Wd['embed.bins.rate.weight'][0]
    if status is not None: wc = wc + Wd['last_status.weight'][status]
    lg = M(wc, 'attr_heads.value_bin') + Wd['embed_transition.value_bin.weight'][codes_t*17 + lb_t]
    p_ = torch.softmax(lg, -1).numpy()
    return (p_[:,1:]*cenmat).sum(1), p_
res_ = []
for gm, ty, st, lay, au in itertools.product(['log','log10'], [3, 0, 1], [None, 0], ['A','B','C'], [1.0, 60.0, 3600.0]):
    pm_, _ = run(gm, ty, st, lay, au)
    res_.append((round(np.mean(np.abs(pm_-tgt)),3), round(np.corrcoef(pm_, tgt)[0,1],4), gm, ty, st, lay, au))
res_.sort(); print(res_[:8]); print(time.time()-t0_)
```