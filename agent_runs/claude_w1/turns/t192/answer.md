The run failed because LNn takes its arguments in a different order than I assumed. I'll print the signatures of LNn, mlp2n and linn, then rerun the same consistency test with the correct order: a spread near 0 across stays for the same code validates q_of.

```python
for fn in ['LNn','mlp2n','linn']: print(fn, inspect.signature(globals()[fn]))
```