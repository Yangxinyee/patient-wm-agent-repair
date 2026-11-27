The states are stored in float32, so precision is not the problem. Sampling z and the GELU variant make no difference. A structural mismatch remains:
- MAP comes out about +2.5 mmHg too high;
- lab glucose about −2 mg/dL too low;
- NLL 1.486 against the tool's 1.4235.

**Hypothesis:** the "last value" register is shared between paired codes. Finger and lab glucose would then update one register, and arterial and cuff MAP another, so a lab glucose at hour 4 would act as the "last value" for a finger reading at hour 5.

**Test:** compute the register from the most recent reading of either paired code, with two choices of whose normalisation and bins to use, and compare error and NLL with the same-code register.
- Supported if the error drops sharply, mainly for readings whose partner code was more recent.

```python
pair = {'glucose_finger':'glu','glucose_lab':'glu','map_arterial':'map','map_cuff':'map'}
fa4['grp'] = fa4.event.map(pair)
fa4 = fa4.sort_values(['stay_id','grp','hour','event']).reset_index(drop=True)
# rebuild index-dependent tensors after resort
def prep(df_):
    out_ = {}
    out_['P'] = torch.tensor(df_.prev.values); out_['R']=torch.tensor(df_.row.values); out_['dt']=torch.tensor(df_.dt.values)
    out_['codes']=torch.tensor(df_.code.values); out_['cen']=np.stack([cen[e_] for e_ in df_.event]); out_['tgt']=df_.predicted_mean.values
    out_['ob']=np.array([fb(e_, v2) for e_, v2 in zip(df_.event, df_.observed)])
    return out_
# shared-register last value: most recent earlier reading (strictly earlier hour) of either code in group
rows_ = []
for (s_, g_), d_ in fa4.groupby(['stay_id','grp']):
    hrs_ = d_.hour.values; vals_ = d_.observed.values; evs_ = d_.event.values
    for i_ in range(len(d_)):
        m_ = hrs_ < hrs_[i_]
        if m_.any():
            j_ = np.where(m_)[0][-1]; rows_.append((d_.index[i_], vals_[j_], hrs_[j_], evs_[j_]))
        else: rows_.append((d_.index[i_], np.nan, np.nan, None))
sh = pd.DataFrame(rows_, columns=['ix','sh_val','sh_h','sh_ev']).set_index('ix')
fa4 = fa4.join(sh)
fa4['last'] = fa4.groupby(['stay_id','event'])['observed'].shift(1); fa4['last_h'] = fa4.groupby(['stay_id','event'])['hour'].shift(1)
def reg_feats(df_, mode):
    if mode=='same': v_, h_, ecode = df_['last'], df_['last_h'], df_.event
    else: v_, h_, ecode = df_['sh_val'], df_['sh_h'], (df_.event if mode=='shared_own' else df_.sh_ev.fillna(df_.event))
    has_ = v_.notna().astype(float).values
    zl_ = np.array([zs(e_, x_) if not np.isnan(x_) else 0.0 for e_, x_ in zip(ecode, v_)])
    lb_ = np.array([fb(e_, x_) if not np.isnan(x_) else 0 for e_, x_ in zip(ecode, v_)])
    la_ = np.log1p(((df_.hour - h_)*3600).fillna(0).values/3600.0)/10*has_
    f8 = np.zeros((len(df_),8)); f8[:,0]=has_; f8[:,1]=la_; f8[:,2]=zl_; f8[:,3]=has_
    return f8, torch.tensor(lb_)
def run11(df_, mode, a='gelu'):
    pp = prep(df_); f8, lbt = reg_feats(df_, mode)
    dec_ = torch.exp(-pp['dt'][:,None]*torch.exp(Wd['core.flow.a'])[None,:])
    b_ = OUT[pp['P']] + LN_((HH[pp['P']]*dec_[:,None,:]).mean(1), 'core.out_norm')
    ctx_ = b_ + Ma((torch.log(pp['dt'])/10)[:,None], 'gap', a)
    zmu = Ma(ctx_, 'z_prior', a)[:, :32]
    q_ = LN_(ctx_ + L(zmu, 'z_head') + Wd['type_emb.weight'][1], 'query_norm')
    wc = q_ + Wd['embed.code.weight'][pp['codes']] + L(torch.tensor(f8), 'last_numeric') + Wd['embed.bins.value.weight'][lbt]
    lg = Ma(wc, 'attr_heads.value_bin', a) + Wd['embed_transition.value_bin.weight'][pp['codes']*17 + lbt]
    p_ = torch.softmax(lg, -1).numpy(); pm_ = (p_[:,1:]*pp['cen']).sum(1)
    nl_ = -np.log(p_[np.arange(len(p_)), pp['ob']] + 1e-12)
    return pm_, p_, nl_, wc, q_
for mode in ['same','shared_own','shared_lastcode']:
    pm_, p_, nl_, _, _ = run11(fa4, mode)
    d_ = fa4.assign(ad=pm_-fa4.predicted_mean)
    print(mode, round(float(np.abs(d_.ad).mean()),3), d_.groupby('event').ad.mean().round(2).to_dict(), 'nll', round(float(nl_.mean()),4), 'tool', round(float(fa4.nll.mean()),4))
```