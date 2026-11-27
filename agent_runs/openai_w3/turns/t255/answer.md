The signed impulse test is roughly symmetric for insulin, but norepinephrine saturates and affects other sensors. I’ll prepare an explicit paired-dose fit rather than adjust gains blindly: it should preserve factual updates while reproducing both signs of the record-derived response. Held-out factual and paired-update errors will decide whether it merits a rollout test.

```python
print("=== FIT BASELINE CHECKS ===")
for nm in ['w0_220','b0_220','w2_220','b2_220']:
    p227={'w0_220':'core.prelude.0.ffn.0.weight','b0_220':'core.prelude.0.ffn.0.bias',
          'w2_220':'core.prelude.0.ffn.2.weight','b2_220':'core.prelude.0.ffn.2.bias'}[nm]
    print(nm,'difference from original',float(torch.linalg.vector_norm(tensor153(globals()[nm])-load_original(p227))))
print("ZERO OVERRIDES",[(n,float(torch.linalg.vector_norm(tensor153(x)-load_original(n)))) for n,x in zero169.items()])
print("PAIRED ITEMS",[(str(k),[(type(x).__name__,tuple(x.shape) if hasattr(x,'shape') else None) for x in v])
                     for k,v in list(pairs187.items())[:2]])
print("STANDARDISATION",hm220.shape,hs220.shape,float(np.min(hs220)),float(np.max(hs220)),
      'Xlast',np.unique(X220[:,-1]),'contrastlast',np.unique(contrast_features220[(0,.5)][:,-1]),
      'train/valid',len(tr220),len(va220))
print("ORIGINAL PARTS",[(type(x).__name__,tuple(x.shape) if hasattr(x,'shape') else None) for x in orig_parts220])
print("DIRECTION ALTERNATIVES",[(n,np.linalg.norm(globals()[n],axis=0).round(3).tolist(),
                               np.round(J186@globals()[n],4).tolist()) 
                              for n in ['directions186','directions187']])
changed227=action_vectors87((.5,1.0))
print("ACTION RETURN",type(changed227).__name__,
      [(type(x).__name__,tuple(x.shape) if hasattr(x,'shape') else None) for x in changed227]
      if isinstance(changed227,(tuple,list)) else tuple(changed227.shape))
print("=== RECORD-EVIDENCE NOTEBOOK EXCERPTS ===")
note227=Path('/work/notes.md').read_text().splitlines()
matches227=[s for s in note227 if any(w in s.lower() for w in ['impulse estimate','record-derived','instrument','random',
                                                            'identif','lag coefficient','structural','sensor response'])]
print('\n'.join(matches227[:16])[:2700])
```
