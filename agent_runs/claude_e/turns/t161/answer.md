The architecture confirms the state is built by a transformer-style recurrent core, so editing inside it would shift every head's factual predictions. The value-head pair edit remains the practical route. NE values above 1.0 go through the action encoder's MLP in a combination never seen in training, so whether a state readout extrapolates there can only be checked through dose_edit rollouts.

Next I'll build the NE level-readout pair with my existing fit_ro5 helper. I don't remember its arguments, such as Pp, so I'll check whether the kernel still holds its input history (In / _ih) and look up my earlier calls to fit_ro5, gated3 and perp.

```python
hist = globals().get('In') or globals().get('_ih')
print('history available:', hist is not None, (len(hist) if hist is not None else 0))
if hist is not None:
    for i, cell in enumerate(hist):
        if 'fit_ro5(' in cell or 'ROn4 =' in cell or 'ROn4=' in cell or 'perp(' in cell:
            for l in cell.split('\n'):
                if any(k in l for k in ['fit_ro5(', 'ROn4', 'ROi5', 'perp(', 'Pp', 'Gn2', 'Gi3', 'gvn']):
                    print(i, '|', l.strip()[:200])
```