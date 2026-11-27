The readouts from n3 failed (insulin rate rms 1.85). The carriers were chosen for low variance of x1, but norm3 divides by each token's std and subtracts the mean. So a direction whose raw projection is a large near-constant becomes a ±9% fluctuating term in n3. A proper fix needs carriers with a low second moment after centring, plus whitened readouts. Even then, any carrier-read rate carries a 9% multiplicative error.

A cleaner alternative is to write the rate term itself (δ_I × g × rate) directly into x1 through head 1's value path. That is exact and does not depend on the state, and it is scaled by `prelude_norm` the same way as an FFN-made correction would be. The FFN would then only add the gated expected-rate correction.

The cost is that the original FFN units see the injected vector. So I'll measure, with the exact prelude replica, how the FFN transforms δ_I added to x1 at drug bundles, at realistic and large magnitudes. A nearly linear, modest response would make this design workable.

```python
ffn_w1 = load('core.prelude.0.ffn.0.weight'); ffn_b1 = load('core.prelude.0.ffn.0.bias')
ffn_w2 = load('core.prelude.0.ffn.2.weight'); ffn_b2 = load('core.prelude.0.ffn.2.bias')
w3t, b3t = load('core.prelude.0.norm3.weight'), load('core.prelude.0.norm3.bias')
def ffn_block(x1s):   # x1s (...,192) slot residual -> x2
    n3_ = torch.nn.functional.layer_norm(x1s, (192,), w3t, b3t, 1e-5)
    return x1s + torch.nn.functional.gelu(n3_@ffn_w1.T + ffn_b1)@ffn_w2.T + ffn_b2
x1s0 = r2['x1'][:, :4, :]
print('ffn replica check %.2e' % float((ffn_block(x1s0) - r2['x2'][:, :4, :]).abs().max()))
dIt = torch.tensor(dI, dtype=torch.float32)
base_x2 = ffn_block(x1s0)
for sc in [5.0, 20.0, 60.0, 150.0]:
    dx2 = (ffn_block(x1s0 + sc*dIt) - base_x2)/sc      # per unit
    resp = dx2[iI].reshape(-1,192).numpy()
    lin = resp.mean(0)
    pe = Gm @ lin; pe_sd = (Gm @ resp.T).std(1)
    print('scale %5.0f |dx2|/|d| %.3f cos(dx2,d) %.3f pred effects per unit (F,L,A,C) %s sd %s' % (sc, np.linalg.norm(lin)/np.linalg.norm(dI), lin@dI/np.linalg.norm(lin)/np.linalg.norm(dI), np.round(pe,3), np.round(pe_sd,3)))
```