The projection is a top-level `z_in` linear layer, not under `core.`. I'll use it and rerun the check against `step_parts`' attention weights. Success is a max error below 1e-3.

```python
P['z_in.weight'], P['z_in.bias'] = load('z_in.weight').double(), load('z_in.bias').double()
def prelude_att(Bs, rows_s, EUs, k, ret_tokens=False):
    hp = torch.tensor(SX[rows_s[k-1], 192:960]).double().reshape(4,192); zk = torch.tensor(SX[rows_s[k], 960:992]).double()
    dt = (Bs[k][0]-Bs[k-1][0])*3600.0; h_ = hp*torch.exp(-dt*torch.exp(P['core.flow.a'].reshape(-1)))
    zin = zk @ P['z_in.weight'].T + P['z_in.bias']
    e_, u_ = torch.as_tensor(EUs[k][0]).double().reshape(-1), torch.as_tensor(EUs[k][1]).double().reshape(-1)
    T = torch.cat([h_ + P['core.slots'].reshape(4,192), e_[None], u_[None], zin[None]], 0)
    Tn = LN_(T, P['core.prelude.0.norm1.weight'], P['core.prelude.0.norm1.bias'])
    Q_ = (Tn[:4] @ P['core.prelude.0.attn.q.weight'].T).reshape(4,4,48).transpose(0,1); KV = Tn @ P['core.prelude.0.attn.kv.weight'].T
    K_, V_ = KV[:, :192].reshape(7,4,48).transpose(0,1), KV[:, 192:].reshape(7,4,48).transpose(0,1)
    A_ = torch.softmax(Q_ @ K_.transpose(1,2) / np.sqrt(48), -1)
    return (A_, A_ @ V_, T, Tn) if ret_tokens else (A_, A_ @ V_)
errs = []
for k in range(1, len(B0)):
    A_, O_ = prelude_att(B0, rows0, EU0, k); errs.append(float((A_ - step_parts(k)[4][:, :4, :].double()).abs().max()))
print('max att err', np.max(errs).round(5), 'median', np.median(errs).round(6))
```