Head 3 does attend the e token at non-drug bundles: mean weight 0.46, reaching 1.0. At drug bundles it puts 0.999 on u (lowest 0.91), with nothing on e or z.

So the gate-free channel works only if the measurement encoder's outputs on the channel dims are made equal and constant. Then e tokens cancel exactly in the (jWI − jR) difference, just as slot and z tokens do.

**Full 5-dim design** (jR = 4, jI = 67, jN = 108 are fixed; two more killed dims serve as jWI and jWN):

| Where | Edit |
|---|---|
| Action encoder | two output dims carry decoded insulin and NE rates, zero for the empty set; a reference dim stays zero |
| Measurement encoder | the same three dims pinned to one constant |
| Head 3 | two value channels repurposed to read only the (rate dim − reference) differences |
| Prelude out-projection | those channels write the rates into staging dims |
| `b_in` | copies the staged difference into the memory dims |
| `z_post`, `z_in` | stop reading these dims |

There is no gate, so non-drug bundles give exactly zero by construction. The remaining cost is losing the measurement encoder's content in three dims, plus two head-3 channels.

To test that cost I must re-encode with modified encoder weights. The encoder helpers read a global params dict. This run checks which global `ln`, `mlp` and `lin` use, and confirms the `z_post` input layout and that `z_in` has a bias. I also want the decoder targets to be exactly linear, so I will check what the rate z-score clipping does to the top of the edited range.

```python
for f in ['ln','mlp','lin','LN','mlpP','encode_set','zpost_mean','tf_loop']:
    try: c_=globals()[f].__code__; print(f, c_.co_names[:25], [x for x in c_.co_consts if isinstance(x,str)][:12])
    except Exception as ex: print(f,'ERR',ex)
print('z_post.0', SHAPES['z_post.0.weight'], 'z_in', SHAPES['z_in.weight'], SHAPES.get('z_in.bias'))
num=VOCAB['numeric']; print([ (d['code'],d['attribute'],d['median'],d['iqr']) for d in num if d['code'] in ('insulin','norepinephrine')])
```