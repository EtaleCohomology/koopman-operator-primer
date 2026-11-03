"""クープマン作用素（DMD）で「谷底からの戻りの速さ」を読む（NumPy）
答えが分かっている地形 h(x, y) = (x^2 - 0.5)^2 + y^2 + a*x を下る流れ v = -grad h を使う。
右の谷底の近くで観測したデータだけから、DMD で固有値を推定し、正解（谷底の曲がり方から計算できる値）と比べる。
a を大きくすると、右の谷底は浅くなり、峠と重なって消える（a = 0.544 付近）。"""
import numpy as np
rng = np.random.default_rng(0)
dt, steps = 0.05, 1          # 観測の間隔 dt、1組＝1歩先

def grad_h(p, a):
    x, y = p
    return np.array([4*x*(x**2 - 0.5) + a, 2*y])

def step(p, a, n=20):         # 細かいルンゲ・クッタ法で dt だけ進める（観測されるのは結果だけ）
    h = dt/n
    for _ in range(n):
        f = lambda q: -grad_h(q, a)
        k1 = f(p); k2 = f(p + h/2*k1); k3 = f(p + h/2*k2); k4 = f(p + h*k3)
        p = p + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return p

def right_valley(a):          # 正解：右の谷底の位置（4x^3 - 2x + a = 0 の右側の根）
    r = np.roots([4, 0, -2, a]); r = np.sort(r[np.isreal(r)].real)
    return r[-1]

def affine_dmd(X, Y):
    """アフィン版のDMD：Y ≈ A X + b を最小二乗で求め、A の固有値と固有ベクトルを返す。
    谷底（平衡点）が原点にないので、定数項 b を別に推定する。
    固有値は複素数のまま返す（一般のDMDでは、複素固有値の偏角が振動の周期を表すため）。"""
    n = X.shape[0]
    Xa = np.vstack([X, np.ones(X.shape[1])])
    M = Y @ np.linalg.pinv(Xa)          # [A | b]
    A = M[:, :n]
    return np.linalg.eig(A)

print(f"{'a':>5} | 谷底のx | x方向：正解 / 推定 | y方向：正解 / 推定 | x方向の戻りの半減期：正解 / 推定")
for a in [0.0, 0.2, 0.4, 0.5]:
    x0 = right_valley(a)
    lam_x = 12*x0**2 - 2          # 谷底近傍のx方向の曲率（ヘッセ行列の固有値）＝戻る速さ
    true_x, true_y = np.exp(-dt*lam_x), np.exp(-dt*2.0)
    # 谷底の近くの200個の点から、1歩先を観測（観測ノイズつき）
    P = np.array([x0, 0.0]) + 0.02*rng.standard_normal((200, 2))
    Q = np.array([step(p, a) for p in P])
    X = (P + 1e-4*rng.standard_normal(P.shape)).T; Y = (Q + 1e-4*rng.standard_normal(Q.shape)).T
    ev, V = affine_dmd(X, Y)
    # 各固有値を、固有ベクトルの成分が大きいほうの方向（x または y）に対応づける
    ix = int(np.argmax(np.abs(V[0, :]))); iy = 1 - ix
    est_x, est_y = ev[ix], ev[iy]
    assert abs(est_x.imag) < 1e-12 and abs(est_y.imag) < 1e-12   # この例では理論上、実数の固有値になる
    hl_true = np.log(2)/lam_x; hl_est = -dt*np.log(2)/np.log(abs(est_x))
    print(f"{a:5.2f} | {x0:+.3f}  | {true_x:.4f} / {est_x.real:.4f}    | {true_y:.4f} / {est_y.real:.4f}    | {hl_true:.2f} / {hl_est:.2f}")
