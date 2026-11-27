The previous output was cut short by the long default argument of `run_combo`, and the last `print` never ran. The useful parts:

- **Config:** 4 attention heads, FFN ×4, 1 prelude layer, 2 recurrent layers, 1 coda layer, 1 loop, 4 slots, memory of 128 entries, z of size 32.
- **Embedding:** it uses an `embed.kq` tensor that ARCHITECTURE.md does not mention.
- **Old helpers:** they take an "act" argument, so earlier I searched over activation choices. The variable `best` probably holds the winning combination.

So I have exact encoders for the embedding, the set encoders and z_post, but no core step yet.

Next I will collect the remaining facts without flooding the output:
- the winning activation combination;
- the unit maps and the embedding tensors;
- the shapes of the state arrays (`X`, `BUN`, `row_of`) that I already loaded from forward pass 2;
- the data columns and the contents of the forward output folders.

If `best` shows a clean match, for example GELU for every MLP, I can write a self-contained file, /work/wm.py, with the whole model step and check it against the tool states.

```python
def short(x,n=600):
    s=repr(x); return s[:n]
for nm in ['best','ACTS','ACTS2','um','rum','V_unit','CODE_ID']:
    print(nm,':',short(globals().get(nm)))
print('NUM keys',list(NUM.keys())[:20])
print('BUN',type(BUN),len(BUN), short(list(BUN.keys())[:3] if isinstance(BUN,dict) else None,200))
print('X',type(X), getattr(X,'shape',None)); print('row_of',type(row_of), short(list(row_of.items())[:3] if isinstance(row_of,dict) else row_of,300))
for n in NAMES:
    if n.startswith('embed') or n.startswith('encode'): print(n,SHAPES[n])
print(open('/data/README.md').read()[:2500])
print(tr.columns.tolist()); print(tr.head(4).to_string()[:1200])
for k in ['fwd_1','fwd_2']: print(k,os.listdir('/work/'+k))
```