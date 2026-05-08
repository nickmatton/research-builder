# Reproduction Report

## 1. Overview

This report summarizes the reproduction of the paper's results. The paper reports a headline
test accuracy of **95.2%**. Our reproduction achieved **95.5%** test accuracy.

## 2. Training Summary

| Metric | Value |
|---|---|
| Total Epochs | 100 |
| Final Training Loss | 0.002540 |
| Final Training Accuracy | 100.0% |
| Best Training Accuracy | 100.0% |
| Minimum Training Loss | 0.001706 |
| Learning Rate | 0.001 |

## 3. Training Curves

### Figure 1: Training Loss over Epochs

```
Training Loss

  0.4193 |*                                                           
  0.3895 |                                                            
  0.3596 |                                                            
  0.3298 |                                                            
  0.3000 |                                                            
  0.2702 |                                                            
  0.2403 |                                                            
  0.2105 |                                                            
  0.1807 | *                                                          
  0.1508 |  *                                                         
  0.1210 |   ***                                                      
  0.0912 |       ***                                                  
  0.0614 |      *   *** *                                             
  0.0315 |             * ***********    *    *     *                  
  0.0017 |____________________________________________________________
         |____________________________________________________________
          Epoch 1                                              Epoch 100
```

### Figure 2: Training Accuracy over Epochs

```
Training Accuracy

  1.0000 |                   *      **** **** ************************
  0.9866 |             * **** ******    *    *                        
  0.9731 |      *** *** *                                             
  0.9597 |     *   *                                                  
  0.9463 |   **                                                       
  0.9329 |  *                                                         
  0.9194 | *                                                          
  0.9060 |                                                            
  0.8926 |                                                            
  0.8791 |                                                            
  0.8657 |                                                            
  0.8523 |                                                            
  0.8389 |                                                            
  0.8254 |                                                            
  0.8120 |____________________________________________________________
         |____________________________________________________________
          Epoch 1                                              Epoch 100
```

## 4. Evaluation Results

### Table 1: Test Set Performance

| Metric | Reproduced | Paper-Reported |
|---|---|---|
| Test Accuracy | 95.5% | 95.2% |
| Test Loss | 0.1946 | — |
| Num Test Samples | 200 | — |
| Num Correct | 191 | — |

### Table 2: Per-Class Accuracy

| Class | Accuracy |
|---|---|
| Class 0 | 93.0% |
| Class 1 | 98.0% |

## 5. Comparison with Paper

| Metric | Paper | Reproduced | Difference | Within ±0.5%? |
|---|---|---|---|---|
| Test Accuracy | 95.2% | 95.5% | +0.3% | ✓ Yes |

### Discrepancy Analysis

The reproduced test accuracy of **95.5%** differs from the paper-reported value of
**95.2%** by **+0.3%**.

This difference is **within** the acceptable tolerance of ±0.5%, indicating a successful reproduction.

Possible sources of minor variation:
- Random seed differences in data splitting and weight initialization
- Minor numerical differences across hardware/software environments
- The paper does not provide full hyperparameter details or exact data splits

### Limitations and Gaps

- **No ablation studies**: The paper does not report ablation studies, so we cannot assess
  individual component contributions.
- **No baseline comparisons**: The paper does not provide baseline results for comparison.
- **Table 1 data missing**: The paper references Table 1 with full benchmark results, but its
  contents were not available from the provided text.

## 6. Conclusion

The reproduction **succeeded**. The reproduced
test accuracy of 95.5% is within the ±0.5% tolerance of the paper-reported 95.2%.
The model trained for 100 epochs, converging to a final training accuracy of
100.0% with a loss of 0.002540.
