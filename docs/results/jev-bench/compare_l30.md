| dataset | metric | Jev | Qwen3.8-27B | jex 0.5B | jex 0.5B+LoRA |
|---|---|---|---|---|---|
| afrixnli | accuracy | **0.733** | 0.700 | 0.467 | 0.500 |
| ag_news † | accuracy | **0.900** | **0.900** | **0.900** | 0.867 |
| agb_de | f1 | **0.000** | **0.000** | **0.000** | **0.000** |
| anli | accuracy | **0.767** | 0.700 | 0.433 | 0.333 |
| arc | accuracy | **1.000** | **1.000** | 0.533 | 0.567 |
| art | accuracy | **0.800** | **0.800** | 0.567 | 0.433 |
| banking77 | accuracy | 0.800 | **0.833** | 0.000 | 0.067 |
| belebele | accuracy | **0.900** | 0.767 | 0.267 | 0.333 |
| bigbench | accuracy | **0.867** | 0.800 | 0.433 | 0.567 |
| boolq | accuracy | 0.933 | **0.967** | 0.533 | 0.600 |
| ceval | accuracy | 0.800 | **0.833** | 0.267 | 0.500 |
| clinc150 | accuracy | **0.867** | 0.833 | 0.000 | 0.100 |
| commonsense_qa | accuracy | **0.800** | **0.800** | 0.533 | 0.500 |
| emotion † | accuracy | **0.700** | **0.700** | 0.633 | **0.700** |
| financial_phrasebank | accuracy | **0.867** | 0.767 | 0.367 | 0.400 |
| go_emotions | macro_f1 | 0.211 | **0.212** | 0.135 | 0.179 |
| hellaswag | accuracy | **1.000** | 0.933 | 0.467 | 0.533 |
| helpsteer2 | mean_spearman | 0.331 | **0.350** | 0.052 | 0.115 |
| imdb | accuracy | **0.967** | **0.967** | 0.833 | 0.800 |
| language_id | accuracy | **1.000** | **1.000** | 0.033 | 0.033 |
| mmlu | accuracy | **0.933** | 0.867 | 0.467 | 0.600 |
| openai_moderation | mean_auprc | **0.723** | 0.568 | 0.215 | 0.417 |
| paws | accuracy | 0.767 | **0.833** | 0.500 | 0.500 |
| prompt_injections | accuracy | **0.833** | **0.833** | 0.600 | 0.700 |
| pubmedqa | accuracy | **0.667** | 0.633 | 0.300 | 0.367 |
| rotten_tomatoes | accuracy | **0.900** | 0.833 | 0.733 | 0.767 |
| sib200 | accuracy | **0.867** | **0.867** | 0.500 | 0.433 |
| sms_spam | f1 | **1.000** | **1.000** | 0.333 | 0.571 |
| sst2 † | accuracy | **1.000** | 0.967 | 0.767 | 0.867 |
| sst5 † | argmax_accuracy | 0.533 | 0.533 | 0.333 | **0.567** |
| stsb | spearman | 0.899 | **0.902** | -0.318 | 0.593 |
| summeval | mean_group_spearman | **0.647** | 0.613 | -0.027 | -0.102 |
| toxic_chat | f1 | 0.500 | **0.667** | 0.000 | 0.000 |
| toxigen | accuracy | **0.933** | 0.867 | 0.600 | 0.600 |
| unfair_tos | micro_f1 | 0.556 | **0.625** | 0.058 | 0.044 |
| winogrande | accuracy | **0.867** | 0.767 | 0.500 | 0.367 |
| **mean, all** (36) | | **0.774** | **0.757** | **0.362** | **0.428** |
| **mean, w/o jex-trained †** (32) | | **0.773** | **0.754** | **0.324** | **0.388** |

Qwen3.8-27B vs Jev: better on 9, equal on 12, worse on 15 of 36

jex 0.5B vs Jev: better on 0, equal on 2, worse on 34 of 36

jex 0.5B+LoRA vs Jev: better on 1, equal on 2, worse on 33 of 36
