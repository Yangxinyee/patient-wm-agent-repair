Neither the slots start state nor the residual posterior fixes it. The best error is still about 3.4.

Next I'll test the order of the three input blocks to z_post. The text says [ctx, e_k, u_k], but the saved weights may use another order. I'll try all 6 permutations with the gap scan, with and without `kq[0]`. A near-zero error would identify the layout.

```python
e_nokq=enc(emb_visit(11,3,0,0)[None],'encode'); e_kq=enc(emb_visit2(11,3)[None],'encode'); u0=torch.zeros(192)
res_=[]
for xg in np.linspace(-3,1,41):
    g_=mlp(torch.tensor([[xg]],dtype=torch.float32),'gap')[0]
    for en,ee in [('nokq',e_nokq),('kq',e_kq)]:
        blocks_={'c':g_,'e':ee,'u':u0}
        for perm in itertools.permutations('ceu'):
            zp=mlp(torch.cat([blocks_[p_] for p_ in perm]),'z_post')[:32]
            res_.append((round(float((zp-zt).abs().max()),4),round(xg,2),en,''.join(perm)))
res_.sort(); print(res_[:5])
```