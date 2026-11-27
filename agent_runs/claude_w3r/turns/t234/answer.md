The encoder rebuild matches `u_of` exactly, but injecting there is ruled out. The encoder norm's scale σ_pre varies by about 70× across action tokens (5.6 to 399, CV 0.98), so the injected gain would vary just as much. The insulin read from mlp2's hidden layer is also mediocre (residual sd 0.13, max 4.9).

I will go back to the ungated linear read from the prelude attention outputs, and make it robust. I will refit it on the recorded tokens plus dose-edited ones. That means recomputing the prelude attention with u replaced by `u_of` of the edited rates. The read must stay linear under the four evaluation edits.

First I check that I can recompute the attention pattern from the stored normalised tokens. Success means near-zero error against the stored `patt` and `HO`.

```python
Wq_=Wt['core.prelude.0.attn.q.weight'].numpy(); Wkv_=Wt['core.prelude.0.attn.kv.weight'].numpy()
Wk_,Wv_=Wkv_[:192],Wkv_[192:]
def prel_heads(N1):            # N1: (B,7,192) norm1 tokens -> attention (B,4,7,7), head outputs (B,4,4,48) for slot queries
    Q=(N1@Wq_.T).reshape(len(N1),7,4,48); K=(N1@Wk_.T).reshape(len(N1),7,4,48); Vv=(N1@Wv_.T).reshape(len(N1),7,4,48)
    L=np.einsum('bihd,bjhd->bhij',Q,K)/np.sqrt(48); L-=L.max(-1,keepdims=True); P_=np.exp(L); P_/=P_.sum(-1,keepdims=True)
    O=np.einsum('bhij,bjhd->bhid',P_[:,:,:4],Vv)
    return P_,O
P_chk,O_chk=prel_heads(n1[:200])
print('patt err %.2e  HO err %.2e (scale %.2f)'%(np.abs(P_chk-patt[:200]).max(), np.abs(O_chk-HO[:200]).max(), np.abs(HO[:200]).max()))
g1n_=Wt['core.prelude.0.norm1.weight'].numpy(); b1n__=Wt['core.prelude.0.norm1.bias'].numpy()
print('norm1 bias max %.3f'%np.abs(b1n__).max(), '| u0 n1 = bias?', np.abs(n1[~act,5][0]-b1n__).max())
```