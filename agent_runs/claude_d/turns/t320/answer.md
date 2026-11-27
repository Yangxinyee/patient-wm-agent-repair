The last output was blocked by the content filter, so I don't know what runs 283–286 did. Two things stand out. The notebook ends at Fwd14, so the results of fwd15 and fwd16 are not recorded. The edited-tensor list shows only the head tensors (`attr_heads.value_bin.*`), so the encoder edit from E5 (`encode_action.mlp1`) seems to have been dropped.

With 14 Python runs left, I will first check the state compactly, with no long listings:
- which saved edit (E3b, E4 or E5enc) matches the current tensors;
- what files fwd13 to fwd16 hold.

If the current tensors match E4.npz exactly and E5enc does not match, that confirms the encoder edit was removed. I then need to find out from fwd15/16 why, before restoring it or not.

```python
import os, json, numpy as np, pandas as pd
print('edited:', edited())
for k in (13,14,15,16):
    d=f'/work/fwd_{k}'
    fs=sorted(os.listdir(d))
    print(f'fwd{k}:', [(f, os.path.getsize(os.path.join(d,f))//1024) for f in fs])
    for f in fs:
        p=os.path.join(d,f)
        if f.endswith('.json'):
            s=open(p).read()
            print('   ', f, s[:400].replace('\n',' '))
        elif f.endswith('.csv'):
            df=pd.read_csv(p)
            print('   ', f, df.shape, list(df.columns)[:10])
for fn in ('E3b.npz','E4.npz','E5enc.npz'):
    z=np.load('/work/'+fn)
    out=[]
    for n in z.files:
        try:
            cur=np.asarray(load(n)); out.append((n[-28:], round(float(np.abs(cur-z[n]).max()),5)))
        except Exception as e:
            out.append((n[-28:], 'err'))
    print(fn, out)
nt=open('/work/notes.md').read(); print('notes len', len(nt), [s in nt for s in ('Fwd15','fwd15','Fwd16','fwd16')])
```