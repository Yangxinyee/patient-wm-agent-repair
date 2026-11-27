The head is extremely sensitive to the view: about 13 to 15 mg/dL per unit, along a consistent direction (cosine 0.93). Moving 0.4 units along it shifts the next glucose prediction by about 5 mg/dL. The view's natural spread along that direction is 1.5 units, so this is a natural glucose-belief axis.

The core update, by contrast, damps a prelude-output injection by roughly 600 times. So the dose effect should enter where leverage is high, through the head path z_prior(ctx) → z_head, a linear map into the view. The core then only needs to carry a clean dose-deviation signal in some direction that z_prior can read. That direction can be quiet rather than glucose-relevant, so SNR matters more than size.

Next I'll look at the core's write path into h: the shapes of c_out, coda and the blocks, the size of softplus(delta_in) and a_bar, and whether z_prior and z_head have spare hidden units. Then I can choose the cleanest route by which a z-token injection leaves a decodable trace in the next view.

```python
for n in NAMES:
    if any(n.startswith(p) for p in ('core.c_out','core.coda','core.out_norm','core.flow','z_prior','z_head','core.blocks.0.0.ffn','core.age')): print(n, SHAPES[n])
sp = torch.nn.functional.softplus(WT['core.delta_in']); ab = torch.exp(-sp*torch.exp(WT['core.a_in']))
print('softplus(delta_in) med %.3f min %.3f max %.3f; a_bar med %.3f min %.3f max %.3f'%(sp.median(), sp.min(), sp.max(), ab.median(), ab.min(), ab.max()))
zp0 = gelu(ctx2.float() @ WT['z_prior.0.weight'].T + WT['z_prior.0.bias']); print('z_prior hidden', zp0.shape[1], 'dead(sd<.01):', int((zp0.std(0)<0.01).sum()))
```