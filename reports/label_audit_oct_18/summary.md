# Label audit (random sample per class)

| class | checked | mislabelled | error rate | 95% CI |
|---|---|---|---|---|
| sweater | 30 | 9 | 30% | 17%-48% |
| cardigan | 30 | 8 | 27% | 14%-44% |
| jacket | 30 | 7 | 23% | 12%-41% |
| hoodie | 30 | 6 | 20% | 10%-37% |
| blazer | 30 | 4 | 13% | 5%-30% |
| coat | 30 | 4 | 13% | 5%-30% |
| t_shirt | 30 | 4 | 13% | 5%-30% |
| tank_top | 30 | 4 | 13% | 5%-30% |
| shirt_blouse | 30 | 3 | 10% | 3%-26% |
| dress | 30 | 0 | 0% | 0%-11% |
| **all** | 300 | 49 | 16.3% | 12.6%-20.9% |

With ~30 images per class the per-class intervals are wide; the overall rate is the more reliable number. A model's measured test accuracy can be at most about 84% if the model never repeats these label errors.
