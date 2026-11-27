/work/head.py holds only the utilities:
- the robust z-score, (x - median)/IQR clipped at ±10;
- the fine-bin function, 1 + the number of edges below x;
- small numpy MLP and LayerNorm helpers.

The value-head computation itself (`head_inputs`, `head_value2`) was defined directly in the kernel during the last session. I need its exact details: which type embedding row, how the last-value age is scaled, what the status input is, and how z is drawn. Next I will try to recover its source with `inspect`, printing only a limited excerpt. If the source is not retrievable, I will rebuild the head from the architecture text and check it against the tool's predicted means in fwd_2.

```python
for fn_name in ['head_inputs', 'head_value2']:
    try:
        src_ = inspect.getsource(globals()[fn_name])
        print('=====', fn_name, len(src_)); print(src_[:2600])
    except Exception as ex:
        print(fn_name, 'no source', type(ex).__name__)
```