My block step reproduces `tf_loop`'s h update, new h and out to float precision (errors up to 6e-5). Age embeddings are added to M before the key/value projection. Together with the exact prelude, I now have a full, readable copy of the core step in /work/lib2.py. That gives every internal quantity the gated write needs.

Next I'll record this in the notebook and classify bundles by their drug atoms using the atom table: insulin start, NE start, END only, or none. Then I'll check which prelude head attends to the u token in each bundle type, per slot. That tells me where to read the insulin and NE rates for the staging write.

```python
with open('/work/notes.md','a') as fh: fh.write('''
- Runs 176-186: KILL COST matters: my kill() in /work/lib1.py (explicit). On 300 train stays (dsE, PO nll 1.630; dsF 1.604):
  old roles8 +0.075!, smallest-rms4 [1,17,4,132] +.002, cheapest-4 [4,67,2,108] +.0023(E)/.0004(F), cheapest-7 +.0055/.0036, 10 dims +.0105.
  Single-dim costs (60 small-rms dims) ~0 +-.0004; cheapest: 4,67,2,108,168,110,33,50,116,17,75,64,49,21. Costs superadditive.
- All measurements at minute 0, all drug starts at minute 5 (separate bundles). Stays ~54 bundles.
- /work/lib2.py: prelude(P,hf,e,u,z) EXACT vs tf_loop store; core_rest(P,hf,E,Mx,Ma,Mm) EXACT (age emb added to M before kv);
  mem_for(ds, srows, e) builds M. tf_loop(P,ds,keep=('aw','E','hf','hn')) -> (H,Out,store incl z).
- PLAN (3 dims/drug): jWI raw staging (prelude attn head3 -> attn.out row), jR shared zero reference, jI memory (abar=1, flow decay);
  gate = 2 GELU units in blocks.0.0 FFN reading (n3_jWI - n3_jR) and a linear gate readout -> write into x_jI;
  readout = 1 value-MLP unit gated to glucose_finger via killed dims (with_code const per code/lastbin), compensation = regression on q.
''')
at_=ds['at']; print(at_.columns.tolist()); print(at_.head(3).to_string()[:400])
```