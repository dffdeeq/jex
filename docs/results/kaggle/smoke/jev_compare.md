| dataset | metric | Jev | Qwen3.8-27B | Gemma-4-E4B | qwen2.5-0.5b-zs |
|---|---|---|---|---|---|
| banking77 | accuracy | **0.800** | 0.600 | 0.600 | 0.000 |
| go_emotions | macro_f1 | **0.119** | 0.077 | 0.095 | 0.095 |
| sst2 † | accuracy | **1.000** | **1.000** | **1.000** | 0.400 |
| **mean, all** (3) | | **0.640** | **0.559** | **0.565** | **0.165** |
| **mean, w/o jex-trained †** (2) | | **0.460** | **0.339** | **0.348** | **0.048** |

Qwen3.8-27B vs Jev: better on 0, equal on 1, worse on 2 of 3

Gemma-4-E4B vs Jev: better on 0, equal on 1, worse on 2 of 3

qwen2.5-0.5b-zs vs Jev: better on 0, equal on 0, worse on 3 of 3
