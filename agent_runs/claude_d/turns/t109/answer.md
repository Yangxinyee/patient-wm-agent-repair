Run 97 gives the code ids: map_cuff 5, glucose_finger 6, map_arterial 7, insulin 8, norepinephrine 9, glucose_lab 10. It also shows atom kinds 0 WORLD, 1 REVEAL (unused) and 2 END. `numeric` is a list, not a dict. I'll print the normalisation entries and the bin centres for the four reading codes. The centres set how much a tilt of the value logits moves the predicted mean.

```python
num=V['numeric']; print(len(num)); 
for e in num: print({k:(v if not isinstance(v,list) else [round(x,2) for x in v]) for k,v in e.items()})
vbc=V['value_bin_centres']; print(type(vbc[0]).__name__, len(vbc))
```