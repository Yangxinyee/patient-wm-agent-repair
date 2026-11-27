The attractor works on paper. Slot queries in head 3 are almost perfectly aligned (median cosine 1.00). An LN-space attractor direction with exactly zero head-3 value gives about 1.0 attention on z at insulin bundles (minimum 0.991) and at nearly all norepinephrine bundles (a few outliers at 0).

Next I'll test the real step, with `step_local` running the true computation, at 30 insulin and 30 norepinephrine bundles. Two checks:
1. **Factual cost.** How much does switching head 3 to the attractor-dominated z token alone move the next glucose and MAP predictions? I need this small, ideally a few tenths.
2. **Channel gain.** For each of the 48 head-3 value directions, injected through z-token directions with zero key and zero mean, how much do the next glucose and MAP predictions move? How consistent is that across bundles?

Strong, consistent sensitivity at small factual cost confirms the design.

```python
Ax = a.clone()  # x-space attractor (zero-mean, unit var)
Cw = torch.cat([Wv3*g1[None], Wk3*g1[None], torch.ones(1,192), Ax[None]],0)   # (98,192)
rhs = torch.zeros(98,48); rhs[:48] = torch.eye(48)
Wm = torch.linalg.pinv(Cw)@rhs        # (192,48): x-space dir per value basis
print('|w_m| med %.2f'%Wm.norm(dim=0).median())
Gg, s_ = 30.0, 0.3; zero = torch.zeros(192)
def scan(ks_):
    imp, J = [], []
    for k in ks_:
        b0 = next_pred(k, ztok_add=zero); bA = next_pred(k, ztok_add=Gg*Ax); imp.append(bA-b0)
        J.append(np.array([(next_pred(k, ztok_add=Gg*Ax + s_*Gg*Wm[:,m]) - bA)/s_ for m in range(48)]))
    return np.array(imp), np.array(J)
for nm_, ks_ in [('ins', ki[::40][:30]), ('ne', kn[::12][:30])]:
    imp, J = scan(ks_)
    gG, gM = J[:,:,0].mean(0), J[:,:,1].mean(0)
    cG = (J[:,:,0]@gG)/(np.linalg.norm(J[:,:,0],axis=1)*np.linalg.norm(gG)); cM = (J[:,:,1]@gM)/(np.linalg.norm(J[:,:,1],axis=1)*np.linalg.norm(gM))
    print(nm_, 'attractor impact |dG| med %.2f p90 %.2f, |dM| med %.2f p90 %.2f'%(np.median(abs(imp[:,0])), np.quantile(abs(imp[:,0]),.9), np.median(abs(imp[:,1])), np.quantile(abs(imp[:,1]),.9)))
    print(nm_, 'sens |gG| %.2f (cos med %.2f) |gM| %.2f (cos med %.2f) cos(gG,gM) %.2f'%(np.linalg.norm(gG), np.median(cG), np.linalg.norm(gM), np.median(cM), gG@gM/np.linalg.norm(gG)/np.linalg.norm(gM)))
    globals()['J_'+nm_] = J
```