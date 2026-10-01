"""VaR backtesting engine: rolling-window violations + Kupiec / Christoffersen tests."""

import numpy as np
from scipy.stats import chi2

Z_SCORES = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.99: 2.3263,
}


def simple_parametric_var(window_returns, confidence=0.95, portfolio_value=1.0):
    """비교용 기본 모델: 단순 표준편차 기반 모수적 VaR."""
    mean = window_returns.mean()
    std = window_returns.std(ddof=1)
    z = Z_SCORES[confidence]
    return (z * std - mean) * portfolio_value


def rolling_backtest(returns, var_func, window=250, confidence=0.95, portfolio_value=1.0, **var_kwargs):
    """매일 과거 window만큼의 데이터로 VaR을 계산하고, 실제 다음날 손실과 비교해 위반 여부를 기록.

    Returns: (var_estimates, violations, actual_losses) — 모두 window일 이후부터의 배열
    """
    returns = np.asarray(returns)
    n = len(returns)
    if n <= window:
        raise ValueError(f"데이터 길이({n})가 window({window})보다 길어야 합니다.")

    var_estimates = np.empty(n - window)
    violations = np.empty(n - window, dtype=bool)
    actual_losses = np.empty(n - window)

    for t in range(window, n):
        hist = returns[t - window : t]
        var_t = var_func(hist, confidence, portfolio_value, **var_kwargs)
        loss_t = -returns[t] * portfolio_value  # 수익률이 음수(손실)면 양수로 변환
        idx = t - window
        var_estimates[idx] = var_t
        violations[idx] = loss_t > var_t
        actual_losses[idx] = loss_t

    return var_estimates, violations, actual_losses


def _safe_log_likelihood(prob, successes, total):
    """0*log(0) 문제를 피하기 위한 이항분포 로그우도 계산."""
    if successes == 0:
        return (total - successes) * np.log(1 - prob)
    if successes == total:
        return successes * np.log(prob)
    return (total - successes) * np.log(1 - prob) + successes * np.log(prob)


def kupiec_test(violations, confidence=0.95):
    """Kupiec's POF(Proportion of Failures) test: 위반비율이 이론치와 통계적으로 같은지 검정."""
    violations = np.asarray(violations)
    n = len(violations)
    x = int(violations.sum())
    p = 1 - confidence
    p_hat = x / n

    ll_null = _safe_log_likelihood(p, x, n)
    ll_alt = _safe_log_likelihood(p_hat, x, n)
    lr_stat = -2 * (ll_null - ll_alt)
    p_value = 1 - chi2.cdf(lr_stat, df=1)

    return {
        "statistic": lr_stat,
        "p_value": p_value,
        "violations": x,
        "total": n,
        "violation_rate": p_hat,
        "expected_rate": p,
        "reject_h0": p_value < 0.05,
    }


def christoffersen_independence_test(violations):
    """위반이 특정 시기에 몰려서(군집) 발생하는지(독립성 위배) 검정."""
    violations = np.asarray(violations).astype(int)
    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(violations[:-1], violations[1:]):
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe_log(x):
        return np.log(x) if x > 0 else 0.0

    ll_restricted = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
    ll_unrestricted = (
        n00 * safe_log(1 - pi01)
        + n01 * safe_log(pi01)
        + n10 * safe_log(1 - pi11)
        + n11 * safe_log(pi11)
    )
    lr_stat = -2 * (ll_restricted - ll_unrestricted)
    p_value = 1 - chi2.cdf(lr_stat, df=1)

    return {
        "statistic": lr_stat,
        "p_value": p_value,
        "pi01": pi01,
        "pi11": pi11,
        "reject_h0": p_value < 0.05,
    }


def conditional_coverage_test(violations, confidence=0.95):
    """Kupiec + Christoffersen 독립성 검정을 합친 종합 검정 (자유도 2)."""
    kupiec = kupiec_test(violations, confidence)
    independence = christoffersen_independence_test(violations)
    lr_stat = kupiec["statistic"] + independence["statistic"]
    p_value = 1 - chi2.cdf(lr_stat, df=2)
    return {"statistic": lr_stat, "p_value": p_value, "reject_h0": p_value < 0.05}


def bad_var_model(window_returns, confidence=0.95, portfolio_value=1.0):
    """데모용 '나쁜' 모델: 변동성을 의도적으로 절반만 반영해 리스크를 과소평가."""
    mean = window_returns.mean()
    std = window_returns.std(ddof=1) * 0.5
    z = Z_SCORES[confidence]
    return (z * std - mean) * portfolio_value


def print_report(name, violations, confidence):
    kupiec = kupiec_test(violations, confidence)
    independence = christoffersen_independence_test(violations)
    cc = conditional_coverage_test(violations, confidence)

    print(f"=== {name} ===")
    print(
        f"위반: {kupiec['violations']}/{kupiec['total']}일 "
        f"(실제 {kupiec['violation_rate']:.2%} vs 기대 {kupiec['expected_rate']:.2%})"
    )
    print(f"Kupiec (비율) 검정       : LR={kupiec['statistic']:.3f}, p-value={kupiec['p_value']:.4f} "
          f"{'→ 기각 (모델 부적합)' if kupiec['reject_h0'] else '→ 기각 못함 (모델 적합)'}")
    print(f"Christoffersen (독립성)  : LR={independence['statistic']:.3f}, p-value={independence['p_value']:.4f} "
          f"{'→ 기각 (위반이 군집됨)' if independence['reject_h0'] else '→ 기각 못함'}")
    print(f"Conditional Coverage(종합): LR={cc['statistic']:.3f}, p-value={cc['p_value']:.4f} "
          f"{'→ 기각 (모델 부적합)' if cc['reject_h0'] else '→ 기각 못함 (모델 적합)'}")
    print()


if __name__ == "__main__":
    # 예시: 변동성 레짐이 몇 차례 바뀌는 3년치(750일) 가짜 수익률 데이터
    rng = np.random.default_rng(42)
    segments = [
        rng.normal(0, 0.010, 250),
        rng.normal(0, 0.025, 100),
        rng.normal(0, 0.010, 250),
        rng.normal(0, 0.030, 50),
        rng.normal(0, 0.012, 100),
    ]
    returns = np.concatenate(segments)
    confidence = 0.95
    window = 250

    _, good_violations, _ = rolling_backtest(returns, simple_parametric_var, window, confidence)
    _, bad_violations, _ = rolling_backtest(returns, bad_var_model, window, confidence)

    print_report("단순 모수적 VaR (정상 모델)", good_violations, confidence)
    print_report("변동성 과소평가 모델 (의도적 결함 모델)", bad_violations, confidence)
