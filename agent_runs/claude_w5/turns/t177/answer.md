The listing was cut off at 60 entries, so I cannot yet tell whether the verified functions (`cstep`, `enc2`, `tf_check`, `probs_from_v0`) and data (`W64`, `bundles_B`, `X_`) survived. The weight names show `b_in`, `c_out`, `prelude.0`, `coda.0` and `blocks.0`.

Next I will:
- check for those names directly;
- list the sub-tensor names under the core blocks, so any new code uses the right keys.

If they are present, I will build the injection from the existing functions. If not, I will rebuild them from the notebook description.

```python
need=['cstep','enc2','tf_check','probs_from_v0','W64','bundles_B','X_','SIDS','H','V0','fixed','TRr','cen','avec','senc']
print({k:(k in globals()) for k in need})
for p in ['core.blocks.0','core.prelude.0','core.coda.0']:
    nn=[n[len(p)+1:] for n in NAMES if n.startswith(p+'.')]
    print(p,len(nn),nn[:30])
print('nblocks',sorted(set(n.split('.')[2] for n in NAMES if n.startswith('core.blocks.'))))
if 'bundles_B' in globals(): 
    b0=bundles_B[list(bundles_B)[0]] if isinstance(bundles_B,dict) else bundles_B[0]
    print(type(bundles_B),len(bundles_B),type(b0), (len(b0) if hasattr(b0,'__len__') else ''))
```