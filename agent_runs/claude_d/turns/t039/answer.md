Scanning the gap input and the small embedding variants gets the error down only to about 2.9. So the convention is still wrong somewhere more structural. The likeliest candidate is the order of the three 192-blocks in z_post's 576-wide input, which I assumed to be [ctx, e_k, u_k].

Next I will try all six block orders for the first bundle, and print the per-block norms of `z_post.0.weight` as a hint. A near-exact match for one ordering would confirm the input layout. If none comes close, the problem is in the context or the encoder.

```python
import itertools
a = atom_vec('hospital_admission', 0, 3)
e1 = set_enc([a],'encode',F.gelu)
ctx = ln(torch.zeros(192),'core.out_norm') + mlp(torch.tensor([0.0]),'gap',F.gelu)
blocks_ = {'ctx':ctx, 'e':e1, 'u':torch.zeros(192)}
Wz = T['z_post.0.weight']
print('z_post.0 block norms', [round(Wz[:,192*i:192*(i+1)].norm().item(),2) for i in range(3)])
for perm in itertools.permutations(['ctx','e','u']):
    zp = mlp(torch.cat([blocks_[p] for p in perm]), 'z_post', F.gelu)
    print(perm, round((zp[:32]-target_z1).abs().max().item(),3), round(np.corrcoef(zp[:32].numpy(), target_z1.numpy())[0,1],4))
```