# VortexKit AI Assistant Skill

## Overview

This skill enables AI assistants to work effectively with the **VortexKit** Python library for high-frequency financial data analysis. VortexKit provides Numba-accelerated implementations of realized measures, jump tests, volatility models, and microstructure analysis tools.

---

## Library Architecture

### Module Organization

```
vortex_kit/
├── realized_measures/    # Variance, covariance, quarticity estimators
├── covariance/           # Extended covariance methods
├── jump_tests/           # Jump detection tests
├── spot_volatility/      # Spot vol, drift estimation
├── har_model/            # HAR volatility models
├── heavy_model/          # HEAVY model
├── lead_lag/             # Lead-lag estimation
├── liquidity/            # Liquidity measures
├── inference/            # Statistical inference
├── kernels/              # Kernel functions
└── utils/                # Utilities (returns, pre-averaging, etc.)
```

### Key Design Principles

1. **NumPy-native**: All functions accept/return NumPy arrays
2. **Numba-accelerated**: Core computations use `@njit(cache=True)`
3. **make_returns pattern**: Functions accepting price data have `make_returns: bool` parameter
4. **Dictionary results**: Tests return `{ztest, pvalue, critical_val, teststat}`
5. **Model objects**: HAR/HEAVY return fitted model dictionaries with `.predict()` methods

---

## Core API Patterns

### Import Patterns

```python
from vortex_kit import rRVar, rBPVar, BNSjumpTest, fit_har

```

### Common Function Signature Pattern

```python
def estimator_name(
    data: np.ndarray,
    *,  # keyword-only arguments after this
    make_returns: bool = False,
    # additional parameters specific to estimator
) -> float | np.ndarray | dict:
    ...
```

### Return Value Patterns

| Function Type | Return Structure |
|--------------|------------------|
| Realized measures | `float` (scalar estimate) |
| Covariance estimators | `float` (scalar) or `np.ndarray` (matrix) |
| Jump tests | `dict` with `ztest`, `pvalue`, `critical_val`, `teststat` |
| Model fits | `dict` with coefficients, fitted values, residuals, predict method |

---

## Realized Measures (realized_measures)

### Variance Estimators

#### rRVar - Realized Variance
```python
from vortex_kit import rRVar

# Basic usage with returns
returns = np.array([0.01, -0.005, 0.015, -0.002])
rv = rRVar(returns)  # Returns: float

# With prices (auto-convert to returns)
prices = np.array([100.0, 101.0, 100.5, 102.0, 101.8])
rv = rRVar(prices, make_returns=True)

# Formula: RV = sum(r_i^2) where r_i = log(P_{i+1}/P_i)
# Reference: Barndorff-Nielsen & Shephard (2004)
```

#### rBPVar - Bipower Variation (Jump-Robust)
```python
from vortex_kit import rBPVar

bpv = rBPVar(prices, make_returns=True)

# Formula: BPV = (pi/2) * sum(|r_i| * |r_{i-1}|)
# Robust to occasional jumps, consistent for integrated variance
# Reference: Barndorff-Nielsen & Shephard (2004)
```

#### rMedRVar - Median Realized Variance
```python
from vortex_kit import rMedRVar

mrv = rMedRVar(prices, make_returns=True)

# Jump-robust using median of 3 consecutive returns
# More robust than BPV to multiple jumps
# Reference: Andersen, Dobrev, Schaumburg (2012)
```

#### rMinRVar - Minimum Realized Variance
```python
from vortex_kit import rMinRVar

min_rv = rMinRVar(prices, make_returns=True)

# Jump-robust using minimum of consecutive absolute returns
# Reference: Andersen, Dobrev, Schaumburg (2012)
```

### Quarticity Estimators

#### rQuar - Realized Quarticity
```python
from vortex_kit import rQuar

rq = rQuar(returns)  # sum(r_i^4)

# Used for asymptotic variance calculations
# Needed for proper jump test standardization
```

#### rTPQuar - Tri-Power Quarticity (Jump-Robust)
```python
from vortex_kit import rTPQuar

tpq = rTPQuar(returns)

# Robust quarticity for jump test inference
# Reference: Barndorff-Nielsen & Shephard (2004)
```

### Noise-Robust Estimators

#### rTSVar / rTSCov - Two-Scale Variance/Covariance
```python
from vortex_kit import rTSVar, rTSCov

# For data with microstructure noise
# Computes RV at two frequencies and combines
# Reference: Zhang, Mykland, Ait-Sahalia (2005)

ts_var = rTSVar(prices, make_returns=True, K=300)  # K = slow scale
```

#### rKernelVar / rKernelCov - Realized Kernel
```python
from vortex_kit import rKernelVar, rKernelCov

# Noise-robust using kernel weighting
# Reference: Barndorff-Nielsen et al. (2008)

kernel_var = rKernelVar(
    prices,
    make_returns=True,
    kernel_type="parzen",  # or "bartlett", "epanechnikov"
    H=300  # bandwidth
)
```

#### rMRVar / rMRCov - Modulated Realized Variance/Covariance
```python
from vortex_kit import rMRVar, rMRCov

# Pre-averaging approach for noisy data
# Reference: Christensen, Kinnebrock, Podolskij (2010)

mrv = rMRVar(prices, make_returns=True, kn=100)  # kn = pre-average window
```

### Noise Estimation

#### ReMeDI - Robust Median Difference Estimator
```python
from vortex_kit import ReMeDI, knChooseReMeDI

# Estimates noise auto-covariance
# Reference: Li & Linton (2021)

# First choose optimal kn
optimal_kn = knChooseReMeDI(prices, make_returns=True, kn_max=50)

# Then compute ReMeDI
noise_cov = ReMeDI(prices, make_returns=True, kn=optimal_kn, lags=1)
```

---

## Covariance Estimators (covariance)

### Hayashi-Yoshida Covariance
```python
from vortex_kit import rHYCov

# For non-synchronous trading data
# Does NOT require time synchronization
# Reference: Hayashi & Yoshida (2005)

hy_cov = rHYCov(
    prices1, times1,
    prices2, times2,
    make_returns=True
)
```

### Threshold Covariance
```python
from vortex_kit import rThresholdCov

# Jump-truncated covariance
# Removes returns exceeding threshold before computing covariance
# Reference: Boudt, Croux, Laurent (2011)
```

### Realized Skewness and Kurtosis
```python
from vortex_kit import rSkew, rKurt

skew = rSkew(returns)  # Realized skewness
kurt = rKurt(returns)  # Realized kurtosis

# Reference: Amaya et al. (2015)
```

---

## Jump Tests (jump_tests)

### BNS Jump Test
```python
from vortex_kit import BNSjumpTest

result = BNSjumpTest(
    prices,
    make_returns=True,
    # Optional: specify realized measures directly
    RV=rv_estimate,
    BPV=bpv_estimate,
    TQ=tri_quarticity
)

# Returns: {
#   'ztest': float,       # z-statistic
#   'pvalue': float,      # p-value
#   'critical_val': float, # 95% critical value (~1.96)
#   'teststat': float     # test statistic
# }

# Interpretation: ztest > critical_val indicates significant jump
# Reference: Barndorff-Nielsen & Shephard (2006)
```

### Aït-Sahalia-Jacod Test
```python
from vortex_kit import AJjumpTest

result = AJjumpTest(
    prices,
    make_returns=True,
    p=4,    # power (2, 3, or 4; authors recommend 4)
    k=2     # scale factor (2, 3, or 4; authors recommend 2)
)

# Compares power variation at two time scales
# H0: No jumps
# Reference: Aït-Sahalia & Jacod (2009)
```

### Intraday Jump Test
```python
from vortex_kit import intradayJumpTest

# Lee-Mykland test for intraday jump detection
# Identifies specific times when jumps occurred
# Reference: Lee & Mykland (2008)
```

### Rank Jump Test
```python
from vortex_kit import rankJumpTest

# Li-Todorov-Tauchen rank-based co-jump test
# For multivariate jump detection
# Reference: Li, Todorov, Tauchen (2017)
```

---

## Volatility Models

### HAR Model Family
```python
from vortex_kit import fit_har

# Supported model types:
#   "HAR"    - Basic HAR-RV (Corsi 2009)
#   "HARJ"   - HAR with jump component
#   "HARCJ"  - Continuous/jump decomposition
#   "HARQ"   - HAR with realized quarticity (BPQ 2016)
#   "HARQJ"  - HAR-Q with jumps
#   "CHAR"   - Continuous HAR (uses BPV for RV)
#   "CHARQ"  - Continuous HAR-Q

result = fit_har(
    rv_data,
    model_type="HAR",
    periods=[1, 5, 22],  # Daily, weekly, monthly lags
    h=1                   # Forecast horizon
)

# Returns dict with:
#   'coefficients': array of fitted coefficients
#   'fitted_values': in-sample predictions
#   'residuals': model residuals
#   'forecast': out-of-sample forecast
#   'r_squared': model fit
#   'model_type': model specification
#   'periods': lag structure used
```

### HEAVY Model
```python
from vortex_kit import fit_heavy

# High-frEquency-bAsed VolatilitY (HEAVY) model
# Uses realized measures for conditional variance
# Reference: Shephard & Sheppard (2010)

result = fit_heavy(
    returns,
    rv_data,
    # returns model dict similar to HAR
)
```

---

## Spot Volatility

```python
from vortex_kit import spotVol, spotDrift, driftBursts

# Nonparametric spot volatility estimation
# Reference: Kristensen (2010)

spot_vol = spotVol(
    prices,
    make_returns=True,
    kernel_type="epanechnikov",
    bandwidth=300  # in number of observations
)

# Drift burst detection
# Reference: Christensen, Oomen, Reno (2018)
db_result = driftBursts(prices, make_returns=True)
```

---

## Lead-Lag Analysis

```python
from vortex_kit import leadLag

# Estimate lead-lag relationship between two assets
# Reference: Hoffmann, Rosenbaum, Yoshida (2013)

result = leadLag(
    prices1,
    prices2,
    lags=range(-10, 11),  # Test lags from -10 to +10
    make_returns=True
)

# Returns: {
#   'contrasts': array of contrast function values,
#   'lead_lag_ratio': float (LLR < 1: asset1 leads asset2)
#   'optimal_lag': int (lag with maximum contrast)
# }
```

---

## Liquidity Analysis

```python
from vortex_kit import getLiquidityMeasures, getTradeDirection

# Calculate 23 microstructure liquidity measures
liquidity = getLiquidityMeasures(
    quotes_data,  # bid/ask data
    trades_data   # trade data
)

# Lee-Ready trade direction classification
directions = getTradeDirection(
    trades,
    quotes,
    method="lee_ready"  # or "quote", "tick", "midpoint"
)
# Returns: 1 for buy, -1 for sell
# Reference: Lee & Ready (1991)
```

---

## Common Patterns and Recipes

### Pattern 1: Basic Volatility Estimation
```python
import numpy as np
from vortex_kit import rRVar, rBPVar, BNSjumpTest

def estimate_volatility(prices):
    """Standard volatility estimation with jump test."""
    rv = rRVar(prices, make_returns=True)
    bpv = rBPVar(prices, make_returns=True)
    
    jump_test = BNSjumpTest(prices, make_returns=True)
    has_jump = jump_test['ztest'] > jump_test['critical_val']
    
    # Use jump-robust estimator if jump detected
    volatility = bpv if has_jump else rv
    
    return {
        'rv': rv,
        'bpv': bpv,
        'has_jump': has_jump,
        'jump_pvalue': jump_test['pvalue'],
        'volatility': volatility
    }
```

### Pattern 2: HAR Model Forecasting
```python
from vortex_kit import fit_har

def forecast_volatility(rv_series, model_type="HARQ"):
    """Fit HAR model and generate forecast."""
    model = fit_har(
        rv_series,
        model_type=model_type,
        periods=[1, 5, 22],
        h=1
    )
    
    return {
        'forecast': model['forecast'],
        'r_squared': model['r_squared'],
        'coefficients': model['coefficients']
    }
```

### Pattern 3: Multi-Asset Covariance Matrix
```python
import numpy as np
from vortex_kit import rHYCov

def estimate_covariance_matrix(price_list, time_list):
    """
    Estimate covariance matrix for multiple assets
    with non-synchronous trading.
    """
    n = len(price_list)
    cov_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                cov_matrix[i, i] = rRVar(price_list[i], make_returns=True)
            else:
                cov = rHYCov(
                    price_list[i], time_list[i],
                    price_list[j], time_list[j],
                    make_returns=True
                )
                cov_matrix[i, j] = cov_matrix[j, i] = cov
    
    return cov_matrix
```

### Pattern 4: Intraday Jump Detection
```python
from vortex_kit import intradayJumpTest

def detect_intraday_jumps(prices, threshold_quantile=0.99):
    """
    Detect jump times within a trading day.
    """
    test_result = intradayJumpTest(prices, make_returns=True)
    
    # Extract jump indicators and times
    jump_indicators = test_result['jumps']  # boolean array
    jump_times = test_result['times']       # corresponding timestamps
    
    return jump_times[jump_indicators]
```

---

## Important Considerations

### Data Preprocessing

1. **Clean data first**: Remove obvious errors, zeros, duplicates
2. **Handle non-trading periods**: Remove overnight gaps if analyzing intraday
3. **Time zones**: Ensure consistent timestamps for multi-asset analysis
4. **Synchronization**: For covariance, consider refresh-time synchronization

### Parameter Selection

| Estimator | Key Parameter | Rule of Thumb |
|-----------|--------------|---------------|
| rTSVar/rTSCov | K (slow scale) | K ≈ n^(1/2) where n = sample size |
| rKernelVar | H (bandwidth) | H ≈ c * n^(3/5) for noise-robustness |
| rMRVar/rMRCov | kn (pre-average) | kn ≈ theta * sqrt(n), theta ∈ [0.5, 2] |
| spotVol | bandwidth | Larger = smoother, smaller = more local |

### Performance Notes

1. **First call overhead**: Numba compilation occurs on first call - expect ~1-2s delay
2. **Cache benefits**: Subsequent calls with same argument types are fast
3. **Array layout**: C-contiguous arrays (default) are optimal
4. **Batch processing**: Process multiple days in loops after warm-up

### Accuracy Validation

When implementing new features:
1. Compare against R `highfrequency` reference implementation
2. Test on sample data with known properties
3. Verify asymptotic properties hold in large samples
4. Check edge cases (single observation, all zeros, etc.)

---

## Error Handling

```python
import numpy as np

def safe_realized_measure(prices, estimator_func):
    """Wrapper with validation."""
    # Check inputs
    prices = np.asarray(prices, dtype=np.float64)
    
    if len(prices) < 2:
        raise ValueError("Need at least 2 price observations")
    
    if np.any(prices <= 0):
        raise ValueError("Prices must be positive")
    
    # Call estimator
    try:
        result = estimator_func(prices, make_returns=True)
        return result
    except Exception as e:
        # Handle Numba or computation errors
        raise RuntimeError(f"Estimation failed: {e}")
```

---

## External Resources

- **R highfrequency docs**: https://cran.r-project.org/web/packages/highfrequency/highfrequency.pdf
- **Reference paper**: https://doi.org/10.18637/jss.v104.i08
- **Numba documentation**: https://numba.pydata.org/
- **Realized volatility survey**: https://www.sciencedirect.com/science/article/pii/S0304407612000340
