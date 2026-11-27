The true effects on the 100 Forward-1 tuning stays, computed from the records' causal kernels, are much larger than what the model predicts:

| edit | true | model |
|---|---|---|
| insulin ×2 | −14.3 mg/dL | −1.88 |
| insulin ×½ | +7.15 mg/dL | +0.98 |
| norepinephrine ×2 | +4.08 mmHg | −0.41 (wrong sign) |
| norepinephrine ×½ | −2.04 mmHg | −0.52 |

The insulin ×2 effect builds from −3.5 at hour 13 to about −17 by hour 19. So the edit has to create a strong, lasting dose effect through the core, and for that I need the encoder outputs, which I have failed to reproduce.

I can measure them directly instead. I will make a temporary diagnostic edit of `z_post` that turns the stored "z posterior mean" into an exact copy of the encoder outputs:
- The ctx block is zeroed. The e and u blocks are set to the identity, with a bias of +20, so every activation sits in its linear range.
- The output layer copies hidden units 0 to 31, minus 20, into the z mean.

Measurement bundles have no actions and vice versa, so one states request will reveal dims 0 to 31 of e_k for measurement bundles and of u_k for action bundles. These do not depend on the state at all. That lets me test encoder hypotheses exactly.

This run only builds and saves the edit. I check that it saves correctly and that its arithmetic reproduces the input on a synthetic vector. I will drop it right after the forward pass.

```python
Wd = torch.zeros(192, 576); Wd[:, 192:384] = torch.eye(192); Wd[:, 384:576] = torch.eye(192)
bd = torch.full((192,), 20.0)
W2d = torch.zeros(64, 192); W2d[torch.arange(32), torch.arange(32)] = 1.0
b2d = torch.zeros(64); b2d[:32] = -20.0
for n_, t_ in [('z_post.0.weight', Wd), ('z_post.0.bias', bd), ('z_post.2.weight', W2d), ('z_post.2.bias', b2d)]:
    try:
        save_edit(n_, t_)
    except Exception as ex:
        print('torch failed', ex); save_edit(n_, t_.numpy())
print(edited())
# sanity: with this edit, z = first 32 dims of (e+u) for |x| << 20
xx = torch.randn(5, 192) * 3
hid = F.gelu(F.linear(torch.cat([torch.zeros(5, 192), xx, torch.zeros(5, 192)], 1), load('z_post.0.weight'), load('z_post.0.bias')))
zz_ = F.linear(hid, load('z_post.2.weight'), load('z_post.2.bias'))[:, :32]
print('max err reveal', (zz_ - xx[:, :32]).abs().max().item())
```