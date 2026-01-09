# Stats Module

**Statistical distributions, metrics, and hypothesis tests with proper p-values**

## Status: ✅ COMPLETE (Phase 8 + Phase 9)

## Distributions

| Function | Description |
|----------|-------------|
| `normal_pdf(x, mu, sigma)` | Normal (Gaussian) PDF |
| `normal_cdf(x, mu, sigma)` | Normal CDF using erf |
| `poisson_pmf(k, lam)` | Poisson PMF |
| `binomial_pmf(k, n, p)` | Binomial PMF |
| `gamma_pdf(x, alpha, beta)` | Gamma PDF |

## Summary Metrics

| Function | Description |
|----------|-------------|
| `skew(data)` | Fisher-Pearson skewness |
| `kurtosis(data, fisher=True)` | Kurtosis (excess if fisher=True) |
| `percentile(data, q)` | q-th percentile |
| `median(data)` | Median (50th percentile) |
| `std(data, ddof=0)` | Standard deviation |
| `variance(data, ddof=0)` | Variance |

## Hypothesis Tests

| Function | Description | P-value |
|----------|-------------|---------|
| `ttest_1samp(a, popmean)` | One-sample t-test | ✅ Normal approx |
| `ttest_ind(a, b, equal_var)` | Two-sample t-test | ✅ Normal approx |
| `chisquare(f_obs, f_exp)` | Chi-square test | ✅ Proper (gammainc) |

## Usage

```python
import loom.stats as stats

# Normal distribution
pdf = stats.normal_pdf(0, mu=0, sigma=1)  # ~0.3989

# Percentile
p75 = stats.percentile([1, 2, 3, 4, 5, 6, 7, 8], 75)

# T-test
t, p_val = stats.ttest_1samp([9, 10, 11], 10.0)
if p_val < 0.05:
    print("Significant difference from mean 10")

# Chi-square with proper p-value (Phase 9)
obs = [30, 10, 10, 10]  # Observed frequencies
chi_stat, p_val = stats.chisquare(obs)
print(f"Chi-square: {chi_stat.item():.2f}, p-value: {p_val:.4f}")
# Output: Chi-square: 13.33, p-value: 0.0039
```

## Phase 9 Enhancement: Proper Chi-Square P-Value

The chi-square test now returns a proper p-value using the regularized incomplete gamma function:

```
p-value = 1 - gammainc(df/2, χ²/2)
```

This replaces the previous placeholder implementation.
