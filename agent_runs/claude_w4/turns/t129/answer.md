The opening lines don't give the attention head count, but they point to `vocab.json["config"]` for sizes. The drug-bundle count (2,319) matches XIN3, so my replica inputs line up with the state rows.

Next I'll print the config to get the head count and any other prelude settings. Then I'll build the CPU prelude at drug bundles and measure how much the slots attend to the u_k and z tokens.

```python
print(VOCAB['config'])
```