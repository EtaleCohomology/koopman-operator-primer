"""クープマン作用素の入門デモ（NumPy）
非線形の動き：x(t+1) = 0.9*x(t),  y(t+1) = 0.5*y(t) + 0.3*x(t)^2
「x, y」だけで線形回帰すると当たらないが、「x, y, x^2」と測り方を1つ増やすと、
動きが行列の掛け算（線形）でぴったり書ける。"""
import numpy as np
rng = np.random.default_rng(1)

def step(s):
    x, y = s
    return np.array([0.9*x, 0.5*y + 0.3*x**2])

# 学習データ：ランダムな100個の出発点から、1歩先を観測
S0 = rng.uniform(-2, 2, (100, 2)); S1 = np.array([step(s) for s in S0])

def fit(features):                          # 「今の特徴量」→「1歩先の特徴量」を最小二乗（線形回帰）で当てる
    X = features(S0).T; Y = features(S1).T
    return Y @ np.linalg.pinv(X)

feat2 = lambda S: S                                           # 測り方：x, y の2つ
feat3 = lambda S: np.column_stack([S, S[:, 0]**2])            # 測り方：x, y, x^2 の3つ
K2, K3 = fit(feat2), fit(feat3)
np.set_printoptions(precision=3, suppress=True)
print("測り方を3つにしたときの行列 K:\n", K3)
print("K の固有値:", np.sort(np.linalg.eigvals(K3).real)[::-1])

# 学習に使っていない出発点から、20歩先まで予測して比べる
s = np.array([1.5, -1.0]); f2 = feat2(s[None])[0]; f3 = feat3(s[None])[0]
for t in range(20):
    s = step(s); f2 = K2 @ f2; f3 = K3 @ f3
print(f"20歩先の本当の y = {s[1]:.5f}")
print(f"  x, y だけで予測した y      = {f2[1]:.5f}")
print(f"  x, y, x^2 で予測した y     = {f3[1]:.5f}")
