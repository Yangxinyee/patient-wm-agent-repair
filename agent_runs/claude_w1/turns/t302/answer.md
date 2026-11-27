The run failed only because `res` was overwritten by the residual array in the previous run. The same minimum-variance decoders are stored in DEC2, so I will rebuild and save the edit from those.

```python
vIu=DEC2[('ins','mvdr',-2.8)]; vNu=DEC2[('ne','mvdr',3)]
Le=build(-2.8,3.0,vIu,vNu,save=True)
```