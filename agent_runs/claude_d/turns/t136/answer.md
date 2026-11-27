Slot-major is the right layout and the states are float32, so the layout isn't the problem. One explanation I haven't checked is the CSV itself. Its predictions average only 4 random z draws, so they may carry several mg/dL of sampling noise. My mean-z and 16-draw versions would then both sit about 7 mg/dL from a noisy target while being right in expectation. Below I measure how much a 4-draw average varies across independent draw sets. If that spread is about 7 mg/dL for glucose and about 2 for MAP, my reproduction is essentially exact.

```python
ms=[]
for sd_ in range(6):
    ms.append(predict(FR,typ=1,gap_scale=10.0,nz=4,seed=100+sd_)[0])
ms=np.array(ms); gl_ = FR['event'].str.startswith('glucose').values
d4 = ms - ms.mean(0)
print('sd of 4-draw mean: gluc', np.sqrt((d4[:,gl_]**2).mean()*6/5).round(2), 'map', np.sqrt((d4[:,~gl_]**2).mean()*6/5).round(2))
e4 = ms - FR['predicted_mean'].values
print('rms vs csv per draw-set gluc', np.sqrt((e4[:,gl_]**2).mean(1)).round(2), 'map', np.sqrt((e4[:,~gl_]**2).mean(1)).round(2))
```