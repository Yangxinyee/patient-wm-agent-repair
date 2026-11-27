That ends the CPU‑designed prelude‑residual idea. Even with my partly inaccurate step replica, the chain says an injection into the prelude residual moves predicted finger glucose by only about 0.006 mg/dL per unit of injection norm. That holds even in the best MAP‑neutral direction. About 1.5–2 mg/dL per unit insulin rate would need injections far larger than the residual itself, because `prelude_norm` squashes everything. This also explains the v3 failure: about 0.006 × 22 ≈ 0.1 mg/dL.

The way out is to **amplify after the normalisation**. The recurrent update adds softplus(δ_in)·b_in(E), which is linear in E. So the plan is:
1. Find a direction w in E‑space that is naturally silent, i.e. near zero at all bundles.
2. Have the carrier write a modest, clean signal along w at drug bundles only.
3. Add a rank‑1 term to `core.b_in.weight` that turns w·E into a large state change in a chosen direction v.

Natural noise along w gets amplified too, so w must be very quiet at every bundle type. This run checks I can rebuild prelude inputs for non‑drug bundles exactly enough: slot tokens from decayed states, the z token, the e token from my stored per‑bundle encodings, and the constant no‑action u token. If these match the stored drug‑bundle tokens, I can compute E at all 5218 bundles and look for silent directions next.

```python
print([n for n in NAMES if n.startswith('encode_action') or n.startswith('z_in') or n.startswith('core.b_in') or n.startswith('core.delta_in') or n.startswith('core.a_in')])
SL=Wt('core.slots'); sid=so['stay_id']; hr=so['hour']; d=dix
first=np.r_[True,sid[1:]!=sid[:-1]]; print('drug bundles first-in-stay:',int(first[d].sum()))
dt_d=(hr[d]-hr[d-1])*3600.
htf_chk=xo[d-1,192:960].reshape(-1,4,192).astype(np.float64)*np.exp(-dt_d[:,None,None]*np.exp(fa_))
print('slot tokens max err %.2e'%np.abs(TOKa[:,:4]-(htf_chk+SL)).max(), ' |slot tok| %.1f'%np.linalg.norm(TOKa[:,:4],axis=-1).mean())
zw,zb=Wt('z_in.weight'),Wt('z_in.bias') if 'z_in.bias' in NAMES else None
zt=xo[d,960:992].astype(np.float64)@zw.T+(zb if zb is not None else 0)
print('z token max err %.2e'%np.abs(TOKa[:,6]-zt).max())
ea=A(e_all); print('e token vs e_all[d] max err %.2e'%np.abs(TOKa[:,4]-ea[d]).max(), '|e| %.2f'%np.linalg.norm(TOKa[:,4],axis=-1).mean())
print('u token |u| drug %.2f'%np.linalg.norm(TOKa[:,5],axis=-1).mean())
```