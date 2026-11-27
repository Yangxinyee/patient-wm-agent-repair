Two solid anchors now hold:
- The stored `out` equals `out_norm(mean over slots of h)` to 1e-6, so the layout and norm are right.
- Row 0, the state after the shared hospital-admission bundle, is identical across stays.

`a_bar` is only 0.27–0.47 per step, so in the gated update `h` is mostly rebuilt each step, not carried. That matters for the memory-channel design later.

Next I use row 0 as a clean test case: zero initial state, known single atom, empty or one-entry memory. I search activations (encoder MLPs, gap, z_post), the first-step dt and the ctx form for an exact z match. If one combination gives z error near 1e-6, it fixes the activations without the ctx ambiguity of later steps. Then I can fit the prelude, block and coda pieces against row 0's `out`/`h`.

```python
a0=BUN[stays[0]][0][1]
best0=[]
for a1 in names_a:
  for a2 in names_a:
    ACTMAP.clear(); ACTMAP.update({'encode.mlp1':a1,'encode.mlp2':a2,'encode_action.mlp1':a1,'encode_action.mlp2':a2})
    e0,u0=enc_bundle(a0)
    for ga in names_a:
      ACTMAP['gap']=ga
      for dtc in [1.0,60.0,600.0,3600.0,7200.0,None]:
        gv = mlp(W0,torch.tensor([math.log(dtc)/10 if dtc else 0.0],dtype=T),'gap')
        for cv in ['bias+g','g','zero']:
          ctx = (W0['core.out_norm.bias']+gv) if cv=='bias+g' else (gv if cv=='g' else torch.zeros(192,dtype=T))
          for za in names_a:
            ACTMAP['z_post']=za
            zo=mlp(W0,torch.cat([ctx,e0,u0]),'z_post')
            for off in [0,32]:
              er=(zo[off:off+32]-Xt[0,960:]).abs().max().item()
              best0.append((er,a1,a2,ga,dtc,cv,za,off))
best0.sort(key=lambda r:r[0])
for r in best0[:8]: print(r)
```