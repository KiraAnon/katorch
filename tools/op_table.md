# 算子性能表（MT6897 / 真机实测）

| 组 | 算子/写法 | 时间 | 状态 |
|---|---|---|---|
| binary | ADD | 8.97 ms | ok |
| binary | DIV | 7.98 ms | ok |
| binary | MAXIMUM | 6.85 ms | ok |
| binary | MINIMUM | 8.25 ms | ok |
| binary | MUL | 7.23 ms | ok |
| binary | SUB | 6.91 ms | ok |
| matmul | BMM_KN_adj00 | 49.24 ms | ok |
| matmul | BMM_NK_adj01 | 28.10 ms | ok |
| matmul | FC_1024x2048x8192 | 28.83 ms | ok |
| pattern | ABS_via_NEG_MAX | 7.60 ms | ok |
| pattern | SOFTMAX_unroll5 | 6.94 ms | ok |
| reduce | MEAN_axis1 | 2.91 ms | ok |
| reduce | REDUCE_MAX_axis1 | 2.90 ms | ok |
| reduce | REDUCE_SUM_axis1 | 3.13 ms | ok |
| shape | RESHAPE | 5.13 ms | ok |
| shape | TRANSPOSE_2D | 4.92 ms | ok |
| unary | ABS | 5.60 ms | ok |
| unary | EXP | 5.41 ms | ok |
| unary | LOG | - | unsupported |
| unary | LOGISTIC | 6.76 ms | ok |
| unary | NEG | 5.47 ms | ok |
| unary | RELU | 7.17 ms | ok |
| unary | RSQRT | 5.72 ms | ok |
| unary | SIN | - | unsupported |
| unary | SQRT | 8.09 ms | ok |
| unary | TANH | 7.52 ms | ok |

## 同组对比

### unary
- EXP                          5.41 ms   (1.00x vs 最快)
- NEG                          5.47 ms   (1.01x vs 最快)
- ABS                          5.60 ms   (1.04x vs 最快)
- RSQRT                        5.72 ms   (1.06x vs 最快)
- LOGISTIC                     6.76 ms   (1.25x vs 最快)
- RELU                         7.17 ms   (1.33x vs 最快)
- TANH                         7.52 ms   (1.39x vs 最快)
- SQRT                         8.09 ms   (1.50x vs 最快)

### binary
- MAXIMUM                      6.85 ms   (1.00x vs 最快)
- SUB                          6.91 ms   (1.01x vs 最快)
- MUL                          7.23 ms   (1.06x vs 最快)
- DIV                          7.98 ms   (1.16x vs 最快)
- MINIMUM                      8.25 ms   (1.20x vs 最快)
- ADD                          8.97 ms   (1.31x vs 最快)

### pattern
- SOFTMAX_unroll5              6.94 ms   (1.00x vs 最快)
- ABS_via_NEG_MAX              7.60 ms   (1.10x vs 最快)

### reduce
- REDUCE_MAX_axis1             2.90 ms   (1.00x vs 最快)
- MEAN_axis1                   2.91 ms   (1.00x vs 最快)
- REDUCE_SUM_axis1             3.13 ms   (1.08x vs 最快)

### shape
- TRANSPOSE_2D                 4.92 ms   (1.00x vs 最快)
- RESHAPE                      5.13 ms   (1.04x vs 最快)

### matmul
- BMM_NK_adj01                28.10 ms   (1.00x vs 最快)
- FC_1024x2048x8192           28.83 ms   (1.03x vs 最快)
- BMM_KN_adj00                49.24 ms   (1.75x vs 最快)
