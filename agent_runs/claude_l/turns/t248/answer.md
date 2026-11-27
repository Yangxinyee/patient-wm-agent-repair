The new policy models give held-out R² 0.62 for insulin (residual sd 0.60 U/h) and 0.84 for NE (residual sd 0.13), with similar values on the tuning stays. The insulin figure looks lower than my notes, but it is consistent with the multiplicative noise I recorded: about 0.49 on the log scale corresponds to roughly 0.58–0.62 R² on the linear scale. NE is somewhat worse than before, but usable. These residuals are the decoder targets.

BUN keys are tuning stay ids such as 2020000000, so tuning and train stays can be joined to the hourly tables.

To collect decoder training data on training stays, I need the bundle builder that made BUN. I'll print its source from /work/setup.py and check its signature before using it in the next run's collection.

```python
s_ = open('/work/setup.py').read(); i_ = s_.find('def build_bundles'); print(s_[i_:i_+1400])
import inspect; print(inspect.signature(build_bundles))
```