`rh` is not the dose correction: it is almost identical across arms and is ±36 on two codes, so that test told me little. The useful by-product is a warning: a decoder fitted on factual levels reproduces only about 24% of the arm-to-arm differences.

That points to fitting any head readout directly on the differences between arms at the same stay-hour. I'll test whether that works using `Dw`. I'll compute insulin dose lags for every `Dw` row under its arm and fit ridge decoders on arm differences, for glucose rows of the training-half stays. Then on held-out stays I'll check R² and the mean response per arm.

To read the rows correctly I also need the code ids and the source of `gen_wc` (how `Dw` was generated).

Decision rule:
- **Head-only route:** the mean decoded response is unbiased per arm, with R² of the differences at or above about 0.5.
- **Channel route:** the decoded differences remain poor.

```python
print(code_gen2.strip()[:1400])
cv = VOCAB['codes']; print("codes:", (dict(list(cv.items())[:14]) if isinstance(cv, dict) else cv[:14]))
DG = dose_stack(Rx, armsW, 'insulin'); print("DG", DG.shape)
import inspect
try: print(inspect.signature(lag_abs))
except Exception as e: print(e)
```