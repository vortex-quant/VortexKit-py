# VortexKit-py Documentation

vortex_kit high-frequency financial data analysis library.

## Overview

VortexKit is a Python library for statistical analysis of high-frequency financial data, providing Numba-accelerated implementations of:

- Realized measures (variance, covariance, quarticity)
- Jump tests (BNS, AJ, intraday, rank co-jump)
- Spot volatility and drift estimation
- HAR and HEAVY models
- Lead-lag estimation
- Liquidity measures
- Inference tools

## Module Structure

```
vortex_kit/
├── realized_measures/    - 22 realized measure estimators
├── covariance/          - 7 extended covariance estimators
├── jump_tests/          - 4 jump detection tests
├── spot_volatility/     - Spot vol, drift, drift-burst
├── har_model/           - HAR model family
├── heavy_model/         - HEAVY model
├── lead_lag/            - Lead-lag estimation
├── liquidity/           - Liquidity measures & trade direction
├── inference/           - IV inference
├── kernels/             - Kernel functions
└── utils/               - Low-level utilities
```

## Quick Reference

### Realized Measures

| Function | Description | Reference |
|----------|-------------|-----------|
| `rRVar` | Realized variance | BNS 2004 |
| `rBPVar` | Bipower variation (jump-robust) | BNS 2004 |
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

## Usage Examples

See individual module documentation for detailed usage.

## References

Key academic references for the implemented methods:

- **BNS 2004**: Barndorff-Nielsen & Shephard, "Power and bipower variation..."
- **ADS 2012**: Andersen, Dobrev & Schaumburg, "Jump-robust volatility..."
- **ZMA 2005**: Zhang, Mykland & Aït-Sahalia, "A tale of two time scales..."
- **HY 2005**: Hayashi & Yoshida, "On covariance estimation..."
- **Corsi 2009**: "A simple approximate long-memory model..."
- **SS 2010**: Shephard & Sheppard, "Realising the future..."
