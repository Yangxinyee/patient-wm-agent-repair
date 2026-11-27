Reorienting after the restart. The last run (227) checked which internal representation carries a counterfactual dose change linearly. It compared each representation under the recorded dose and under a changed dose. A ridge decoder fitted on factual rows tracked the true dose change well from the encoder output `u` (slope 0.88). It tracked it poorly from the core output `out` (slope 0.54). That means the core state at prediction time carries the dose change only weakly, so any edit has to get the dose signal into the glucose prediction another way.

I have no edits saved yet and 73 Python runs left, so I need to move to a concrete edit soon. First I'll recover my full notes (only the tail was shown) and see which scripts exist and what they contain. That way I can reuse the validated local re-implementation (impl/core/head/rollout) instead of rebuilding it.

```python
import os, time, glob
notes = open('/work/notes.md').read()
print(len(notes))
head = notes[:-8000]
print(head[:9000])
print('....')
fl = sorted(glob.glob('/work/*'), key=os.path.getmtime)
for f in fl:
    print(f, os.path.getsize(f), time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(f))))
```