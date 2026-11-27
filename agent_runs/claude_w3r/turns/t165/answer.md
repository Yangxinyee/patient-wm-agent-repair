Two results.
- Head 2's 48-dim value vector still carries the insulin rate well (R² 0.978) and the norepinephrine rate less well (0.909).
- Head 2 attends to the u token only at drug bundles (0.994). At other bundles its weight on u averages 0.075, so it reads other tokens there. A rank-1 read of head 2's output would therefore inject state-dependent noise at non-drug bundles.

To quantify this, I'll reimplement just the prelude attention (tokens = [decayed h + slots, e, u, z_in(z)]). I'll check that it reproduces `step_parts`' attention weights exactly. Then I can fit a read-vector c on head 2's output with target = rate at drug bundles and 0 elsewhere. Matching attention weights (max error < 1e-3) would confirm the reimplementation.

```python
P = {k: load(k).double() for k in NAMES if k.startswith('core.') and ('prelude' in k or k in ('core.slots','core.flow.a') or k.startswith('core.z_in'))}
print([k for k in P if not k.startswith('core.prelude.0')], {k: tuple(P[k].shape) for k in P if 'z_in' in k or k in ('core.slots','core.flow.a')})
def prelude_att(Bs, rows_s, EUs, k):
    hp = torch.tensor(SX[rows_s[k-1], 192:960]).double().reshape(4,192); zk = torch.tensor(SX[rows_s[k], 960:992]).double()
    dt = (Bs[k][0]-Bs[k-1][0])*3600.0; h_ = hp*torch.exp(-dt*torch.exp(P['core.flow.a'].reshape(-1)))
    zin = zk @ P['core.z_in.weight'].T + P['core.z_in.bias'] if 'core.z_in.bias' in P else zk @ P['core.z_in.weight'].T
    e_, u_ = torch.as_tensor(EUs[k][0]).double().reshape(-1), torch.as_tensor(EUs[k][1]).double().reshape(-1)
    T = torch.cat([h_ + P['core.slots'].reshape(4,192), e_[None], u_[None], zin[None]], 0)
    Tn = LN_(T, P['core.prelude.0.norm1.weight'], P['core.prelude.0.norm1.bias'])
    Q_ = (Tn[:4] @ P['core.prelude.0.attn.q.weight'].T).reshape(4,4,48).transpose(0,1); KV = Tn @ P['core.prelude.0.attn.kv.weight'].T
    K_, V_ = KV[:, :192].reshape(7,4,48).transpose(0,1), KV[:, 192:].reshape(7,4,48).transpose(0,1)
    A_ = torch.softmax(Q_ @ K_.transpose(1,2) / np.sqrt(48), -1)       # heads x 4 x 7
    return A_, (A_ @ V_)                                               # heads x 4 x 48
B0, rows0, EU0 = stay_data(int(sorted(recs.stay_id.unique())[0]))
errs = []
for k in range(1, len(B0)):
    A_, O_ = prelude_att(B0, rows0, EU0, k); errs.append(float((A_ - step_parts(k)[4][:, :4, :].double()).abs().max()))
print('max att err', np.max(errs).round(5), 'median', np.median(errs).round(6))
```