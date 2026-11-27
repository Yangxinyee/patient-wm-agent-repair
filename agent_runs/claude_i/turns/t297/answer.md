The token sensitivities agree well across bundles: the length of the mean sensitivity is about 85% of the average per-bundle length. But the gain is small, about 0.27 mg/dL glucose per unit of token norm.

That gain is too small for the size of injection needed. A lag-1 effect of about −2 mg/dL per unit/h of deviation, over deviations of several units/h, would require token perturbations of 20–70 units. That is far outside the token's normal range and probably nonlinear.

Before choosing a design I want three numbers:
1. how linear the glucose shift stays as the token injection along the best glucose-only direction grows (norms 2–40);
2. how strongly the head's glucose and MAP predictions respond to a shift of the head's input view, and whether glucose and MAP directions are separable;
3. how many dead hidden units the z_prior network has, since that network only feeds the head.

If the view gain is much larger and z_prior has spare units, I'll use a two-stage route: a small clean token signal carried in the state, amplified on the head side through z_prior and z_head. If the token response is linear up to about 40, the direct route is enough.

```python
Jg = (Jtok[:64,:,0]+Jtok[:64,:,1]).mean(0)/2; Jmp = (Jtok[64:,:,2]+Jtok[64:,:,3]).mean(0)/2
v = Jg - (Jg@Jmp)/(Jmp@Jmp)*Jmp; vh = torch.as_tensor(v/np.linalg.norm(v), dtype=dtw)
res = []
for k in kI[:20]:
    H=[];O=[]
    for a in (0,2,5,10,20,40):
        h1,o1 = one_step(k, a*vh); H.append(h1); O.append(o1)
    P = preds_next(torch.stack(H), torch.stack(O), k); res.append(P-P[0])
res = np.array(res); print('token norm 2,5,10,20,40 -> mean shift [gf gl ma mc]'); print(res.mean(0)[1:].round(2))
JV_ = []
for k in kI[:20]:
    h1,o1 = one_step(k); dtn=float(dts[k+1]); lv = lastvals(sid3[k], float(hr3[k+1]))
    ctx = ctx_of_state(o1.double()[None].repeat(385,1), h1.double()[None].repeat(385,1,1), np.full(385,dtn))
    ev = torch.zeros(385,192,dtype=torch.float64); ev[:192] = 0.25*torch.eye(192,dtype=torch.float64); ev[192:384] = -0.25*torch.eye(192,dtype=torch.float64)
    Pv = np.stack([N(value_mean(ctx,[c]*385,np.full(385,lv[ci][0]),np.full(385,lv[ci][1]),extra_view=ev)[0]) for ci,c in enumerate(codes4)],1)
    JV_.append((Pv[:192]-Pv[192:384])/0.5)
JV_ = np.array(JV_); print('view J: |mean| per output', np.linalg.norm(JV_.mean(0),axis=0).round(2), 'mean|J_k|', np.linalg.norm(JV_,axis=1).mean(0).round(2))
gm = JV_.mean(0); print('cos(gf,gl) %.2f cos(gf,ma) %.2f cos(ma,mc) %.2f' % tuple(np.dot(gm[:,a],gm[:,b])/np.linalg.norm(gm[:,a])/np.linalg.norm(gm[:,b]) for a,b in ((0,1),(0,2),(2,3))))
Wp0 = N(load('z_prior.0.weight')); bp0 = N(load('z_prior.0.bias')); prp = N(C2)@Wp0.T + bp0
print('z_prior hidden', Wp0.shape, 'dead', int((gel(prp).std(0)<0.01).sum()), 'z_head', tuple(SHAPES['z_head.weight']))
```