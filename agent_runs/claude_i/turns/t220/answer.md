The readout from the head's input fails. Even with large random dose edits:
- the insulin deviation at lag 1 has a held-out R² of only 0.22, and lower at later lags;
- norepinephrine is about 0.03;
- on factual readings the readout's error is as large as the deviation itself.

So the normalised state the head sees hardly records the dose history. That fits the original model's weak dose response. A readout added to the head alone cannot work. The dose has to be written into the state on purpose, and the only free place to compute it is the dead `z_post` units, which see the drug encoding directly.

The earlier derivative check only showed that the existing head ignores z changes. What matters now is whether a z change at an insulin bundle leaves a consistent, readable trace in the head input at the next reading and the one after.

This run measures, for each z direction at 100 insulin bundles, how the head input changes at the next two readings. From that I compute, for the best direction:
- the signal-to-noise ratio against the natural spread of the head input;
- how consistent the response is across bundles;
- how much of it persists to the second reading.

The route is viable if, with a z injection of order 1, the readout noise is below about 0.2 units/h. If that needs a z far outside its normal range of about 0.2, the route fails.

```python
def stepk(k, h_prev, o_prev, dz=None):
    hf = h_prev.to(fa_.dtype)*torch.exp(-dts[k].to(fa_.dtype)*torch.exp(fa_))
    cx = ctx_from(o_prev.to(fa_.dtype), hf, dts[k]); z0 = zpost_mean(cx, Ee[k].to(fa_.dtype), Uu[k].to(fa_.dtype))
    if dz is not None: z0 = z0 + dz
    j0 = int(np.where(sid3==sid3[k])[0][0]); im = np.arange(max(j0,k-127), k+1); ag = torch.as_tensor((tmin[k]-tmin[im])*60.0, dtype=fa_.dtype)
    return core_step(hf, Ee[k].to(fa_.dtype), Uu[k].to(fa_.dtype), z0, Ee[im].to(fa_.dtype), ag, V)
def q_after(h, o, k1): return q_of(ctx_of_state(o.double()[None], h.double()[None], float(dts[k1])).double())[0]
isread = lambda k: any(c in INFO[k] for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff'])
okQ = [k for k in okI if k+3 < NB and sid3[k+3]==sid3[k] and isread(k+3)]
sQ = np.random.RandomState(2).choice(okQ, 100, replace=False); eps=0.2; D1=[]; D3=[]
for k in sQ:
    k=int(k); out1=[]; out3=[]
    for j in range(-1,32):
        dz = None if j<0 else torch.zeros(32, dtype=fa_.dtype).index_fill_(0, torch.tensor([j]), eps)
        h1,o1 = stepk(k, HHt[k-1], OUTt[k-1], dz); q1 = q_after(h1,o1,k+1)
        h2,o2 = stepk(k+1, h1, o1); h3,o3 = stepk(k+2, h2, o2); q3 = q_after(h3,o3,k+3)
        out1.append(q1.numpy()); out3.append(q3.numpy())
    out1=np.array(out1); out3=np.array(out3); D1.append((out1[1:]-out1[0])/eps); D3.append((out3[1:]-out3[0])/eps)
D1=np.array(D1); D3=np.array(D3)   # (100,32,192)
Sq = np.cov(QQ.T) + 1e-4*np.eye(192); Si = np.linalg.inv(Sq)
M1 = D1.mean(0); G = M1 @ Si @ M1.T; ev_, evec = np.linalg.eigh(G); v = evec[:,-1]
w_read = Si @ (v @ M1); sig1 = D1 @ v @ w_read; sig3 = D3 @ v @ w_read; noise = np.sqrt(w_read @ Sq @ w_read)
print('top SNR eig', np.round(ev_[-3:],3), '| signal per unit z: mean', round(float(sig1.mean()),3), 'sd', round(float(sig1.std()),3), '| noise sd', round(float(noise),3))
print('z amp needed for 0.2 u/h noise:', round(float(noise/(0.2*sig1.mean())),2), '| persistence q3/q1', round(float(sig3.mean()/sig1.mean()),3), '| natural z sd ~0.21')
```