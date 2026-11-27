The dose_edit file gives one row per stay-hour and edit, with the shift, its standard error over the 20 paths, and a `reached` flag. `true_shift(D, G, edit, dvec, drug)` takes those rows, the tuning grid and an impulse-response vector.

Only 29 of these stays overlap sel1, so I need fresh truth estimates for this set. I will apply my record-based impulse responses to the recorded doses on exactly these rows:
- for insulin, the finger-based, lab-based and averaged versions (d_f, d_l, d_avg);
- for norepinephrine, the candidates dN, dN2 and dN_use.

I will first print those vectors to confirm what each is. The spread across versions gives a rough uncertainty band, and the size of the remaining gap for each readout sets the next step.

```python
for nm in ['d_f','d_l','d_avg','dN','dN2','dN_use','TGf','TGl','TI','TG2','TI2']:
    if nm in globals(): print(nm, np.round(np.asarray(globals()[nm],dtype=float),3))
res_t={}
for ed,dr,vecs in [('insulin_x2','ins',['d_f','d_l','d_avg']),('insulin_half','ins',['d_f','d_l','d_avg']),('ne_x2','ne',['dN','dN2','dN_use']),('ne_half','ne',['dN','dN2','dN_use'])]:
    for vn in vecs:
        try:
            t_=true_shift(de,Gtu,ed,np.asarray(globals()[vn],dtype=float),dr)
            val=float(np.nanmean(t_)) if hasattr(t_,'__len__') else float(t_)
            res_t[(ed,vn)]=val; print(ed,vn,'truth est %.3f'%val, type(t_).__name__, (len(t_) if hasattr(t_,'__len__') else ''))
        except Exception as ex: print(ed,vn,'ERR',ex)
```