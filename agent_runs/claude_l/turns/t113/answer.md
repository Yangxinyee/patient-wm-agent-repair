Most of the kernel variables are replica-debugging leftovers. The data-analysis objects (policy model, residual tables) are gone, so I'll rebuild a compact per-stay-hour table.

The test I need comes before any edit design:
- **Question:** when the model predicts glucose with the record visible, is its error tied to the dose residual only, or to the full dose?
- **Residual only:** the model has folded the predictable part of dosing into how it reads glucose. Simply strengthening its dose pathway would then damage factual fit, and the edit must act on dose-minus-expected-dose.
- **Full dose:** amplifying the dose pathway would also improve factual fit.

First I need the data layout to build the table.

```python
print(open('/data/README.md').read()[:2500])
import pandas as pd
print(tu.shape, list(tu.columns))
print(tu.head(12).to_string()[:1800])
print({c['name']:c['id'] for c in VOCAB['codes']})
```