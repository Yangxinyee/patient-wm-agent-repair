The notebook is updated. To send the dose_edit request I need the 100 tuning-stay ids in the request itself, so next I print them as a JSON list. They are the same stays used in forward pass 1 and in the g-estimation targets (ins x2 -14.3, x½ +7.15, NE x2 +4.11, x½ -2.05).

```python
print(json.dumps([int(x) for x in sel_A]))
```