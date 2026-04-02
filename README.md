# VortexKit-py

[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![Numba](https://img.shields.io/badge/accelerated-numba-green)](https://numba.pydata.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**VortexKit** is a high-performance Python library for statistical analysis of high-frequency financial data. It provides Numba-accelerated implementations of functions.

## Overview

VortexKit is a Python reimplementation of the algorithms from the R [`highfrequency`](https://cran.r-project.org/web/packages/highfrequency/) package (Boudt, Cornelissen, Kleen, Payseur, Sjoerup et al.), designed for use in quantitative finance. Every algorithm has been carefully validated against the reference R implementation to ensure numerical accuracy.

### Key Features

- **Realized Measures**: multiple estimators for variance, covariance, quarticity, semivariance
- **Jump Tests**: Statistical tests for price jump detection (BNS, AJ, intraday, rank)
- **Volatility Models**: HAR family and HEAVY model implementation
- **Spot Volatility**: Nonparametric spot volatility and drift estimation
- **Lead-Lag**: Lead-lag estimation via Hayashi-Yoshida contrast functions
- **Liquidity Analysis**: microstructure liquidity measures + trade direction classification
- **Inference Tools**: Standard errors and confidence bands for integrated variance

All functions accept and return **NumPy** arrays, making them compatible with any DataFrame library (Polars, Pandas, etc.). Internally, Numba JIT compilation ensures near-C performance.

## AI Assistant Skill

This repository includes an **AI Assistant Skill** for working with VortexKit. See [`agent_skill/SKILL.md`](agent_skill/SKILL.md) for documentation on using this library with AI coding assistants.


## Quick Start

```python
import numpy as np
from vortex_kit import rRVar, rBPVar, BNSjumpTest, fit_har

# Realized variance & bipower variation (jump-robust)
prices = np.array([100.0, 101.0, 100.5, 102.0, 101.5])
rv = rRVar(prices, make_returns=True)      # Realized variance
bpv = rBPVar(prices, make_returns=True)    # Bipower variation

# Jump test
result = BNSjumpTest(prices, make_returns=True)
print(f"z = {result['ztest']:.4f}, p = {result['pvalue']:.4f}")

# HAR model for volatility forecasting
# See module documentation for detailed examples
```

## Module Structure

```
vortex_kit/
├── realized_measures/    - 22 realized measure estimators
├── covariance/           - 7 extended covariance estimators
├── jump_tests/           - 4 jump detection tests
├── spot_volatility/      - Spot vol, drift, drift-burst
├── har_model/            - HAR model family
├── heavy_model/          - HEAVY model
├── lead_lag/             - Lead-lag estimation
├── liquidity/            - Liquidity measures & trade direction
├── inference/            - IV inference
├── kernels/              - Kernel functions
└── utils/                - Low-level utilities
```

## API Reference

### Realized Measures

| Function | Description | Reference |
|----------|-------------|-----------|
| `rRVar` | Realized variance | BNS 2004 |
| `rBPVar` | Bipower variation (jump-robust) | BNS 2004 |
| `rBPCov` | Bipower covariance | BNS 2004 |
| `rMedRVar` | Median realized variance | ADS 2012 |
| `rMinRVar` | Minimum realized variance | ADS 2012 |
| `rSVar` | Realized semivariance | - |
| `rSemiCov` | Realized semicovariance | - |
| `rQuar` | Realized quarticity | BNS 2004 |
| `rTPQuar` | Tri-power quarticity | BNS 2004 |
| `rMedRQuar` | Median realized quarticity | - |
| `rMinRQuar` | Minimum realized quarticity | - |
| `rQPVar` | Quad-power variation | - |
| `rMPVar` | Multipower variation | - |
| `rCov` | Realized covariance | - |
| `rTSVar` | Two-scale realized variance | ZMA 2005 |
| `rTSCov` | Two-scale realized covariance | ZMA 2005 |
| `rKernelVar` | Realized kernel variance | BHLS 2008 |
| `rKernelCov` | Realized kernel covariance | BHLS 2008 |
| `rMRVar` | Modulated realized variance | CKP 2010 |
| `rMRCov` | Modulated realized covariance | CKP 2010 |
| `ReMeDI` | Noise auto-covariance estimator | LL 2021 |
| `knChooseReMeDI` | Optimal kn selection | LL 2021 |

### Extended Covariance

| Function | Description | Reference |
|----------|-------------|-----------|
| `rHYCov` | Hayashi-Yoshida covariance | HY 2005 |
| `rThresholdCov` | Threshold covariance | BCZL 2011 |
| `rAVGCov` | Subsampled covariance | ZMA 2005 |
| `rOWCov` | Outlyingness-weighted covariance | BZ 2010 |
| `rRTSCov` | Robust two-scale covariance | BZ 2010 |
| `rSkew` | Realized skewness | AT 2015 |
| `rKurt` | Realized kurtosis | AT 2015 |

### Jump Tests

| Function | Description | Reference |
|----------|-------------|-----------|
| `BNSjumpTest` | BNS ratio test | BNS 2006 |
| `AJjumpTest` | Aït-Sahalia-Jacod test | AJ 2009 |
| `intradayJumpTest` | Lee-Mykland intraday test | LM 2008 |
| `rankJumpTest` | Li-Todorov-Tauchen rank test | LTT 2017 |

### Models

| Function | Description | Reference |
|----------|-------------|-----------|
| `fit_har` | HAR family models | Corsi 2009 |
| `fit_heavy` | HEAVY model | SS 2010 |

### Spot Volatility

| Function | Description | Reference |
|----------|-------------|-----------|
| `spotVol` | Nonparametric spot volatility | Kristensen 2010 |
| `spotDrift` | Spot drift estimation | - |
| `driftBursts` | Drift burst detection | COR 2018 |

### Lead-Lag & Liquidity

| Function | Description | Reference |
|----------|-------------|-----------|
| `leadLag` | Lead-lag estimation | HRY 2013 |
| `getLiquidityMeasures` | 23 liquidity measures | - |
| `getTradeDirection` | Lee-Ready classifier | LR 1991 |

## Reference Implementation

This library is based on the R [`highfrequency`](https://cran.r-project.org/web/packages/highfrequency/) package:

- **Package**: `highfrequency` - Tools for Highfrequency Data Analysis
- **CRAN**: https://cran.r-project.org/web/packages/highfrequency/
- **GitHub**: https://github.com/jonathancornelissen/highfrequency


## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use VortexKit in your research, please cite this Python implementation

## Disclaimer

This software is provided for research and educational purposes. Always validate results against the reference implementation for critical applications. The authors are not responsible for any financial losses incurred from using this software.
