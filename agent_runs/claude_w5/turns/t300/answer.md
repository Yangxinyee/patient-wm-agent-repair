The z-sensitivity hypothesis is refuted. At hour 13 the spread across z draws is the same for the edited and original models (3.7 mg/dL). My local edited and original predictions are also close to each other.

The decisive cases are stays like 2020000054. There the tool's original model matches my local prediction (175.6 vs 175.2), but the tool's edited model gives 160.3 against my 175.6. Both factual paths were teacher-forced through 12:05, and on these very stays the tool's edited core matches mine to 2e-6 in states mode.

So the dose_edit rollout must process the record up to hour 12 differently from the states/factual mode. The likeliest candidate is a random core z, drawn from the posterior per path rather than taken as its mean. My edit would be hypersensitive to that, because the stage-1 kv edit gives the z token (and the slots and e) value vectors of 4e3–2e4. These leak through head 1 wherever attention to u is not ~100%, which is the case at reading bundles. The original model barely cares.

I'll read the core section of the documentation for how z is used.

```python
print('\n'.join(f'{i}: {l}' for i, l in enumerate(L_) if 0 <= i <= 50))
```