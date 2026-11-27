Under GELU the head rebuild works best with the time flow on, type_emb row 1 and gap input log(dt in seconds)/10. That matches the architecture text for the view. The error is still 2.58 units, though, and is biased mostly upward on glucose_finger.

The remaining suspect is the layout of the 8 `last_numeric` inputs. Fitting that layout on only 50 readings of one stay risks overfitting. A factual pass on all 99 stays in my saved states would give 5,500 readings, enough to fit and check the head rebuild properly. That rebuild then supports both the search for unused units and offline checks of the factual cost. The same pass also gives the model's prediction errors, which I need to estimate how much dose effect the model is missing.

Last time I mistakenly sent placeholder ids, so this run prints the exact 99-stay list for the request.

```python
import json
print(json.dumps([int(s_) for s_ in sel_states]))
print(len(sel_states), set(sel_states) == set(int(s_) for s_ in st.stay.unique()))
```