The prelude normalizes the transported signal and then passes it through another nonlinear layer, so large corrections may saturate. I’ll measure the direction norms and signed impulse responses, and recover the paired-dose feature arrays. Saturation or unequal signed responses would argue against simply increasing a gain.

```python
print("READOUT ORDER", codes171)
print("DIRECTION NORMS", np.linalg.norm(directions187,axis=0).round(4))
print("CALIBRATED LINEAR RESPONSE", np.round(J186@directions187,5))
print("RATE SUMMARY",np.round(np.quantile(rates187,[0,.25,.5,.75,.95,1],axis=0),4))
print("OVERRIDE KEYS",list(zero169))
print("REFIT FORMAT",type(refit220[1.0]), 
      {k:(tuple(v.shape) if hasattr(v,'shape') else type(v).__name__) for k,v in refit220[1.0].items()}
      if isinstance(refit220[1.0],dict) else None)
print("CONTRAST FEATURE FORMAT",
      {str(k):(type(v).__name__,tuple(v.shape) if hasattr(v,'shape') else len(v)) 
       for k,v in contrast_features220.items()})
print("PAIR FORMAT",
      {str(k):(type(v).__name__,tuple(v.shape) if hasattr(v,'shape') else len(v)) 
       for k,v in list(pairs187.items())[:3]})
print("=== ACTION ARRAY HELPER ===")
print(logic224(action_vectors87)[:2000])
print("CONTEXT OBJECTS",
      {n:tuple(globals()[n].shape) for n in ['C154','E154','U154','zin154','zb154','tokens206',
                                            'orig_raw220','current_raw220','H220','R220'] 
       if n in globals()})
impulses226=np.stack([directions187[:,j]*s for j in range(2) for s in [-2,-1,0,1,2]])
r226=response186(impulses226)
print("SIGNED IMPULSES (drug, amplitude, change)")
for j in range(2):
    b226=r226[5*j+2]
    for k,s in enumerate([-2,-1,0,1,2]):
        print(j,s,np.round(r226[5*j+k]-b226,4).tolist())
```
