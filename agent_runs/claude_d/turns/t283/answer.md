No unit in `encode_action.mlp1` is fully dead: every unit's GELU output reaches at least 0.14 on some action atom. So repurposing a unit means removing its contribution and putting its mean back through the output bias. The cost is what remains.

This run ranks units by (activation spread over all action atoms) × (output-weight norm), using factual and doubled ranges for both drugs. For the top candidates it computes the exact change in the action encoding u after ablation with mean compensation. In the same run I solve the three input directions:
- **Final gate:** NE START ∧ bin 16.
- **Probe gate:** any NE START.
- **z_rate reader.**

All three are constrained exactly orthogonal to the numeric, unit, kind and other categorical embeddings, so the gates cannot drift with dose z-scores. A candidate unit with negligible ablation cost (Δu ≪ 0.01) confirms the plan is free.

```python
INSU=[aIN(r,u,ru) for r in np.linspace(0.01,17.5,60) for u in range(6) for ru in range(4)]
NEA=[aNE(r) for r in np.linspace(0.01,1.0,40)]+[aNE(r) for r in np.linspace(1.0,2.1,20)]
ENDS=[aEND(c,u,ru) for c in (8,9) for u in range(6) for ru in range(4)]
AT=np.array(INSU+NEA+ENDS); act=gel(AT@W10.T+b10); W12=Wt('encode_action.mlp1.2.weight')
score=act.std(0)*np.linalg.norm(W12,axis=0); cand=np.argsort(score)[:25]
sub=np.array(NEA[::2]+INSU[::24]+ENDS[::4]); base=np.array([uenc2([a]) for a in sub])
res=[]
for j in cand:
    c=act[:,j].mean(); W0m=W10.copy(); W0m[j]=0; b0m=b10.copy(); b0m[j]=-50.0; b2m=Wt('encode_action.mlp1.2.bias')+W12[:,j]*c
    uu=np.array([uenc2([a],W0=W0m,b0=b0m,b2=b2m) for a in sub]); res.append((int(j),float(score[j]),float(np.abs(uu-base).max()),float(np.abs(uu-base).mean())))
res.sort(key=lambda t:t[2]); print('ablation (unit,score,max|du|,mean|du|):', [(a,round(b,4),round(c,4),round(d,5)) for a,b,c,d in res[:8]])
CONS=[C_[8],C_[9]]+list(K_)+list(EK_)+list(Br)+list(Bd)+[Bv[0]]+list(Wn.T)+[bn]+list(UN)+list(RU)+[Wt(n)[i] for n in ['embed.categorical.action.weight','embed.categorical.route.weight','embed.categorical.status.weight','embed.kq.weight','embed.text.weight'] for i in range(SHAPES[n][0])]
Mc=np.array(CONS); nK,nE=len(K_),len(EK_)
def tgt(code9,code8,kind0,kind2,br16,brlo,bd16,bdlo,zr=0.0):
    t=[code8,code9]+[kind0,0,kind2]+[0]*nE+[brlo]*16+[br16]+[bdlo]*16+[bd16]+[0]+[0,0,0,0,zr,0]+[0]+[0]*(len(UN)+len(RU))+[0]*(len(CONS)-2-nK-nE-34-1-6-1-len(UN)-len(RU))
    return np.array(t,float)
wGf=np.linalg.lstsq(Mc,tgt(0,-2,0,-2,.5,-.5,.5,-.5),rcond=None)[0]; wGp=np.linalg.lstsq(Mc,tgt(1,-1,0,-2,0,0,0,0),rcond=None)[0]; wZ=np.linalg.lstsq(Mc,tgt(0,0,0,0,0,0,0,0,1.0),rcond=None)[0]
for nm,w,t in [('Gf',wGf,tgt(0,-2,0,-2,.5,-.5,.5,-.5)),('Gp',wGp,tgt(1,-1,0,-2,0,0,0,0)),('Z',wZ,tgt(0,0,0,0,0,0,0,0,1.0))]: print(nm,'|w|',np.linalg.norm(w).round(2),'fit err',np.abs(Mc@w-t).max().round(6))
print('Gf on NE r=.9,1.2,2; ins r=8,17:', [round(float(aNE(r)@wGf),3) for r in (.9,1.2,2.0)], [round(float(aIN(r,0,0)@wGf),3) for r in (8,17)])
```