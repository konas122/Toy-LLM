# Llama

## RMSNorm

$$y_i = \frac{x_i}{\text{RMS}(x)} \gamma_i, \quad \text{where} \quad \text{RMS}(x) = \sqrt{\epsilon + \frac{1}{n} \sum x_i^2}$$

$$
\text{silu}(x) = x \cdot \sigma(x), \quad \text{where} \quad \sigma(x) \text{ is the logistic sigmoid.}
$$
