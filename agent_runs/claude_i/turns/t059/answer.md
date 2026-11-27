The head replica is validated. Averaged over z draws, the per-code mean differences are 0.2–0.37, the slopes 1.02–1.04, and the per-event NLLs match closely: glucose_finger 1.4489 vs 1.4493, map_arterial 1.3807 vs 1.3800. So I can compute exactly, offline, how any head-only edit changes the teacher-forced value NLL, because states don't change in teacher forcing.

The plan is a head-level correction. Each glucose prediction gets a tilted distribution whose mean shifts by δ = Δc₁·dev_t + Δc₂·dev_t−1 + …, with dev = dose − E[dose | decision reading]. I can implement this only if the head's 192-dim input q or its hidden layer linearly encodes the recent dose, the decision reading, and hence the deviations.

Next, I will measure that decodability on the 100 stays' glucose predictions with cross-validated ridge regressions. Targets: D_t, D_t−1, R_t, dev_t, dev_t−1. Out-of-sample R² above about 0.8 for D_t and dev_t makes the approach viable. Low R² for dev, despite good R² for D and R, means the step function E[D|R] is the hard part.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, GroupKFold
codes_ = np.array([CID[e] for e in evs])
# hidden activations & q (prior-mean z)
view_ = out_ + m_ + gv_ + zp_[:,:32] @ P['zhW'].T + P['zhb']
q_ = lnorm(view_ + P['te'][1], P['qw'], P['qb'])
def hidden_of(F_, q):
    hasl = (~np.isnan(F_.lastv.values)).astype(float); age = np.where(hasl>0, (F_.hour.values - F_.lasth.values)*3600, 0.0)
    zl = np.zeros(len(F_)); lb = np.zeros(len(F_), int); ev_ = F_.event.values
    for ev in np.unique(ev_):
        ix = (ev_==ev)&(hasl>0); zl[ix] = zsc(ev,'value',F_.lastv.values[ix]); lb[ix] = fbin(ev,'value',F_.lastv.values[ix])
    feats = np.zeros((len(F_),8)); feats[:,0]=hasl; feats[:,3]=np.log1p(age)/10; feats[:,6]=zl
    cd = np.array([CID[e] for e in ev_])
    wc = q + P['ce'][cd] + feats @ P['LW'].T + P['Lb'] + P['bv'][lb] + P['br'][0] + P['ls'][0]
    return np.maximum(wc @ Tn('attr_heads.value_bin.0.weight').T + Tn('attr_heads.value_bin.0.bias'), 0), wc
hid_, wc_ = hidden_of(F, q_)
F['h_']=F.hour.round().astype(int)
F['D1']=[Dd.get((s,h-1),np.nan) for s,h in zip(F.stay_id,F.h_)]; F['D2']=[Dd.get((s,h-2),np.nan) for s,h in zip(F.stay_id,F.h_)]
F['dev1']=[Jd.get((s,h-1),np.nan) for s,h in zip(F.stay_id,F.h_)]; F['dev2']=[Jd.get((s,h-2),np.nan) for s,h in zip(F.stay_id,F.h_)]
Rr = J['R']; F['R1']=[Rr.get((s,h-1),np.nan) for s,h in zip(F.stay_id,F.h_)]
gix = np.where(F.event.isin(['glucose_finger','glucose_lab']).values & F.D1.notna().values & F.D2.notna().values & F.R1.notna().values)[0]
grp = F.stay_id.values[gix]
for nmX, Xf in [('q', q_[gix]), ('hidden', hid_[gix]), ('state960', X3[F.pk.values[gix], :960])]:
    line=[]
    for tg in ['D1','D2','R1','dev1','dev2']:
        yy = F[tg].values[gix]
        pr = cross_val_predict(RidgeCV(alphas=np.logspace(-2,4,13)), Xf, yy, groups=grp, cv=GroupKFold(5))
        line.append(f"{tg}:{1-np.var(yy-pr)/np.var(yy):.3f}")
    print(nmX, len(gix), " ".join(line))
```