Using the top Jacobian direction, injection strength α = 2 moves u by 26% of its norm and α = 4 by 58%. The response is somewhat superlinear. I will use α = 2.5 at q = 1. With q = 0.5 + 0.5·rate spanning [0.5, 1], the probe also tests how linear the injected signal is once it reaches the head.

This run:
- builds the probe in unit 674 (gate κ = 14 on "any NE START", pre-activation 4.38 + 1.77·z_rate so the unit stays in GELU's linear regime, original mean folded into the bias),
- checks gate leakage on insulin and END atoms offline (should be about 0),
- checks the injected size and bf16 pre-activation error,
- saves the probe on top of E4,
- picks the 100 train stays (from my 400 with saved states) that have the most NE records, for a "states both" pass.

```python
import copy
j=674; kap=14.0; al=2.5
W0n=W10.copy(); b0n=b10.copy(); W2n=W12.copy(); b2n=Wt('encode_action.mlp1.2.bias').copy()
c674=float(act[:,j].mean()); b2n+=W12[:,j]*c674
W0n[j]=kap*wGp+1.77*wZ; b0n[j]=-kap+4.38; W2n[:,j]=al*v1/6.0
pre=lambda A_: A_@W0n[j]+b0n[j]
print('NE START act', [round(float(gel(pre(aNE(r)))),3) for r in (.01,.3,.6,1.0)], 'max |act| ins', float(np.abs(gel(pre(np.array(INSU)))).max()), 'END', float(np.abs(gel(pre(np.array(ENDS)))).max()))
bf=lambda x: torch.tensor(x,dtype=torch.float32).to(torch.bfloat16).to(torch.float32).numpy()
pb=np.array([float(bf(W0n[j])@bf(aNE(r))+bf(b0n[j:j+1])[0]) for r in np.linspace(.01,1,30)]); pf=np.array([pre(aNE(r)) for r in np.linspace(.01,1,30)]); print('bf16 pre err max', np.abs(pb-pf).max().round(4))
du=[np.linalg.norm(uenc2([aNE(r),aE],W0=W0n,b0=b0n,W2=W2n,b2=b2n)-uenc2([aNE(r),aE])) for r in (.1,.5,1.0)]; print('|du| NE bundles r=.1,.5,1:', np.round(du,2))
dui=max(np.abs(uenc2([a],W0=W0n,b0=b0n,W2=W2n,b2=b2n)-uenc2([a])).max() for a in INSU[::7]+ENDS); print('max|du| insulin/END', round(dui,5))
save_edit('encode_action.mlp1.0.weight',torch.tensor(W0n,dtype=torch.float32)); save_edit('encode_action.mlp1.0.bias',torch.tensor(b0n,dtype=torch.float32)); save_edit('encode_action.mlp1.2.weight',torch.tensor(W2n,dtype=torch.float32)); save_edit('encode_action.mlp1.2.bias',torch.tensor(b2n,dtype=torch.float32))
S400=json.load(open('/work/selS.json'))+json.load(open('/work/selT.json'))
TR=pd.read_csv('/data/train.csv',usecols=['stay_id','event']); cnt=TR[TR.event=='norepinephrine'].groupby('stay_id').size()
selP=[int(s) for s in sorted(S400,key=lambda s:-cnt.get(s,0))[:100]]; json.dump(selP,open('/work/selP.json','w')); print('NE recs in selP', int(cnt.reindex(selP).fillna(0).sum()), edited()); print(json.dumps(selP))
```