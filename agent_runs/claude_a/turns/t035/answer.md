Run 31 listed all 181 tensors and the config. They match the architecture text: d = 192, 4 attention heads, 4 slots, z of 32, one prelude layer, two recurrent layers and one coda layer. Every MLP is a two-layer Sequential, so its activation function is still unknown. `last_numeric` takes 8 inputs, not the 4 the text lists, so some details of how atoms are built are missing from the description.

I conclude that any dose signal must be written into the recurrent state while the action bundle is processed, and then be read by the value head at later glucose and MAP predictions. To design an edit like that safely, I need a CPU reimplementation of the model, checked against the `states` output I already have: out, the 4 h slots and the z posterior mean for each bundle.

Next I will look at the vocabulary's atom fields and normalisation, the states file I saved, the kernel variables from my unfinished reimplementation, and one stay's raw events. That will let me pin down how bundles are built. If out equals out_norm applied to the mean of the 4 slots in the states file, my reading of the state layout is confirmed. If it does not, the layout differs from the text and I will rethink.

```python
import numpy as np, torch
print('codes', VOCAB['codes'])
print('event_kinds', VOCAB['event_kinds'])
print('atom_kinds', VOCAB['atom_kinds'])
print('numeric_note', VOCAB['numeric_note'])
print('value_bin_note', VOCAB['value_bin_note'])
print('numeric', json.dumps(VOCAB['numeric'])[:2500])
print('centres keys', (list(VOCAB['value_bin_centres'].keys()) if isinstance(VOCAB['value_bin_centres'], dict) else len(VOCAB['value_bin_centres'])))
print('S files', S.files if hasattr(S,'files') else None)
for kk in (S.files if hasattr(S,'files') else []):
    print(kk, S[kk].shape, S[kk].dtype, S[kk][:3] if S[kk].ndim==1 else '')
print('W keys', list(W.keys())[:20], 'Wt', list(Wt.keys())[:10], 'ACT', list(ACT.keys())[:10])
print('fa', fa.shape, 'fr cols', list(fr.columns)[:20], fr.shape)
# out vs out_norm(mean slots)
st_ = S['states'] if 'states' in S.files else S[S.files[0]]
print(st_.shape)
o = torch.tensor(st_[:5, :192]); hh = torch.tensor(st_[:5, 192:960]).reshape(5,4,192)
on = torch.nn.functional.layer_norm(hh.mean(1), (192,), load('core.out_norm.weight'), load('core.out_norm.bias'), eps=1e-5)
print('out vs out_norm(mean slots) maxabs', (o-on).abs().max().item(), o.abs().mean().item())
sid0 = tr.stay_id.iloc[0]
print(tr[tr.stay_id==sid0].head(25).to_string())
```