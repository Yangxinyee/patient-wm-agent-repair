Keeping the zero memory entries for action-only bundles is still the best option, at out-error 0.052. I'll accept this fidelity for now; any edit I design will be checked with the forward tool anyway.

The dose signal can reach the recurrent state only through the prelude attention from the four slot tokens to the action token u_k (or the z token). So the key question is whether some attention head reliably puts high weight on u_k at insulin and norepinephrine bundles. If one head does, its value channel is a clean route for injecting a dose signal.

```python
def prelude_attn(h_p, e_, u_, z_, dt_, Pp):
    hf = h_p*torch.exp(-dt_*torch.exp(Pp['core.flow.a']))
    T_ = torch.cat([hf + Pp['core.slots'], e_[None], u_[None], lin_(z_[None], 'z_in', Pp)])
    xn = ln_(T_, 'core.prelude.0.norm1', Pp)
    q = (xn @ Pp['core.prelude.0.attn.q.weight'].T).reshape(-1,4,48).transpose(0,1)
    k = (xn @ Pp['core.prelude.0.attn.kv.weight'].T)[:, :192].reshape(-1,4,48).transpose(0,1)
    return torch.softmax(q @ k.transpose(1,2)/np.sqrt(48), -1)  # (head, query, key)
AW = {'ins': [], 'ne': [], 'meas': []}
for s_ in stays12:
    idx_ = np.where(sid4 == s_)[0]; bl_ = stay_bundles(s_)
    mins = [g_.minute.iloc[0] for g_ in bl_]
    for k_ in range(1, len(bl_)):
        g_ = bl_[k_]; e_, u_, ha, hm = enc_bundle(g_)
        a_ = prelude_attn(X4d[idx_[k_-1]][192:960].reshape(4,192), e_, u_, X4d[idx_[k_], 960:992], (mins[k_]-mins[k_-1])*60.0, P0)
        key_ = 'ins' if (g_.code == 8).any() else ('ne' if (g_.code == 9).any() else 'meas')
        AW[key_].append(a_[:, :4, :].numpy())
for key_, L_ in AW.items():
    A_ = np.stack(L_)  # n, head, slot, token
    print(key_, len(L_), 'mean attn to u per head (avg slots):', np.round(A_[:, :, :, 5].mean((0,2)), 3), ' to z:', np.round(A_[:, :, :, 6].mean((0,2)), 3), ' to e:', np.round(A_[:, :, :, 4].mean((0,2)), 3))
    print('   min over bundles of attn to u (head max over slots-avg):', np.round(A_[:, :, :, 5].mean(2).min(0), 3))
```