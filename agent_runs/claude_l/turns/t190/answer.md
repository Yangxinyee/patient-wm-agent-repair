Functions created inside earlier runs have no retrievable source. However, `setup.py` and `wm.py` are on disk and hold the bundle builder and set encoders. The kernel has 100 tuning stays already as bundles (`BUN`), their exact states (`Xs`), and the train/tuning id lists.

To run the teacher-forced replica on training stays I need `build_bundles`' signature and the code-id mapping. I'll print both files and `CODE_ID`.

I expect to be able to build bundles for any train stay. That would let me fit, from the model's own context vectors, how the policy dose is predicted, and measure how the model's prediction errors relate to past dose residuals. If `build_bundles` only works on preloaded data, I'll need to rebuild that step.

```python
print(open('/work/setup.py').read())
print("=====")
print(open('/work/wm.py').read()[:4100])
print("=====")
print(CODE_ID)
print(open('/data/README.md').read()[:2500])
```