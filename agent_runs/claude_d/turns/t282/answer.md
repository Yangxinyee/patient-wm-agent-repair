The u(r) diagnostic changes the picture. In-range NE dose is coded almost entirely by the fine bins: the projection is a step function (0.89 / 0.25 / 0.88) rather than a smooth trend. Above rate 1.0 the encoding moves off the in-range manifold (residual 1.5 → 5.0), and capping the bin only partly helps (0.7 → 4.1).

So the model has no representation for "more NE than the recorded maximum", and an encoder-only fix cannot express it. My plan is a dedicated out-of-range channel:
1. Gated `encode_action.mlp1` units fire only for NE START atoms in bin 16 and inject a known signal proportional to rate/2, the recorded dose before doubling.
2. A gated value-head unit at map_arterial rows reads that signal and shifts MAP.

Bin-16 NE atoms never occur in the records, and x1/2 never leaves the range. So this channel has no factual cost and affects only the NE x2 readout.

To learn how the injection appears at the head, I first need a probe: a temporary injection on all recorded NE START atoms, then one "states" pass. This run prepares it:
- find mlp1 hidden units that are dead for every insulin/NE atom (factual and doubled ranges, all unit-index combinations),
- solve gate directions that are exactly invariant to the numeric z features.

```python
ins={d['attribute']:d for d in VOCAB['numeric'] if d['code']=='insulin'}; print({k:(v['median'],v['iqr'],v['fine_bin_edges'][-3:]) for k,v in ins.items()})
ti=TU[TU.event=='insulin']; print('ins dose/rate', (ti.dose/ti.rate).describe()[['mean','min','max']].round(3).to_dict(), ti.rate.max(), ti.dose_unit.unique(), ti.rate_unit.unique())
rI=float((ti.dose/ti.rate).median())
def aIN(r,u,ru): return C_[8]+K_[0]+EK_[2]+Br[bb(r,ins['rate'])]+Bd[bb(rI*r,ins['dose'])]+Bv[0]+Wn@np.array([0,0,zz(rI*r,ins['dose']),1,zz(r,ins['rate']),1])+bn+UN[u]+RU[ru]
def aEND(c,u,ru): return C_[c]+K_[2]+EK_[2]+Br[0]+Bd[0]+Bv[0]+bn+UN[u]+RU[ru]
rin=np.r_[np.linspace(0.01,2*ti.rate.max()*1.1,120)]; rne=np.linspace(0.01,2.1,80)
ATS=[aNE(r) for r in rne]+[aIN(r,u,ru) for r in rin for u in range(6) for ru in range(4)]+[aEND(c,u,ru) for c in (8,9) for u in range(6) for ru in range(4)]
ATS=np.array(ATS); W10,b10=Wt('encode_action.mlp1.0.weight'),Wt('encode_action.mlp1.0.bias')
Hp=ATS@W10.T+b10; Ha=np.abs(gel(Hp)).max(0); o=np.argsort(Ha); print('deadest units', [(int(i),float(Ha[i])) for i in o[:10]])
print('n units max|act|<1e-3:', int((Ha<1e-3).sum()), ' <1e-2:', int((Ha<1e-2).sum()))
```