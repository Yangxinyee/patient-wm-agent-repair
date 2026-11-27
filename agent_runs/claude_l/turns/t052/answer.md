These leftovers show the previous attempt never closed the gap. Both the z-matching and the state-matching searches had errors of order 0.5 to 1.6 with no clear winner. The conventions it settled on were never recorded, so I am restarting the reimplementation from scratch.

This time I will work incrementally and save code to /work so it survives restarts. I start with checks that need no guessing:
1. Is `out` exactly `out_norm(mean over slots of h)` in the states?
2. Is the first bundle of every stay (hospital admission, identical atoms) given the same state and the same z across stays? If so, it depends only on constants and the initial state. That makes it a clean target for pinning down the encoders, the activations and the initial conditions one piece at a time.

If (1) holds to about 1e-5 and (2) shows identical rows, I have firm anchors to build from. If not, my reading of the architecture is already off at the output.

```python
X2 = torch.tensor(S2['x']); sidv = S2['stay_id']; hrv = S2['hour'].astype(float); nat = S2['n_atoms']
hS = X2[:, 192:960].reshape(-1, 4, 192); outS = X2[:, :192]; zS = X2[:, 960:]
def LNt(v, w_, b_, eps=1e-5):
    return F.layer_norm(v, (v.shape[-1],), w_, b_, eps)
on_w, on_b = load('core.out_norm.weight'), load('core.out_norm.bias')
rec_ = LNt(hS.mean(1), on_w, on_b)
print('out vs out_norm(mean h): max abs err', (rec_ - outS).abs().max().item(), 'mean abs', (rec_ - outS).abs().mean().item())
first = np.array([np.where(sidv == s_)[0][0] for s_ in batch1])
print('first bundle hours', np.unique(hrv[first]), 'n_atoms', np.unique(nat[first]))
zf = zS[first]; hf = hS[first]
print('z first: spread across stays (max std over dims)', zf.std(0).max().item(), ' h first spread', hf.std(0).max().item())
print('z first[0][:8]', zf[0, :8])
print('h first norms per slot', hf[0].norm(dim=1))
sec = first + 1
print('second bundle hours', np.unique(hrv[sec])[:5], 'z2 spread', zS[sec].std(0).max().item())
print('z std over all bundles per dim:', np.round(zS.std(0).numpy(), 3))
```