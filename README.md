[English](./README.md) | [日本語](./README.ja.md)

# Koopman Operator Primer Demos (Python)

Two small demos that check, on examples with known answers, the Koopman operator (treating nonlinear dynamics linearly by adding more observables) and dynamic mode decomposition (DMD, which fits a matrix to data).

## Related articles (Japanese)

- Qiita (for engineers who have forgotten high-school math): "数学のクープマン作用素は経営データ分析に活用できるか" (https://qiita.com/etale_cohomology/items/2066b6bbfed3c6311db8)
- Zenn (for readers with advanced math background): same title (URL to be added after publication)

## Files

### `koopman_beginner_demo.py`

Uses the nonlinear dynamics x(t+1) = 0.9 x(t), y(t+1) = 0.5 y(t) + 0.3 x(t)^2.

- With only x and y as observables, linear regression cannot predict the next state.
- Adding x^2 (observables x, y, x^2) makes the dynamics exactly linear.
- The fitted matrix has eigenvalues 0.9, 0.81 and 0.5, matching the true values.

### `koopman_dmd_early_warning.py`

Uses the gradient flow down the known landscape h(x, y) = (x^2 − 0.5)^2 + y^2 + a·x.

- Estimates eigenvalues by affine DMD (with a separately estimated constant term), using only data observed near the right valley bottom.
- Compares them with the true values computed from the curvature at the valley bottom.
- As the tilt a increases and the valley flattens, the eigenvalue approaches 1 and the recovery half-life grows (the right valley merges with the saddle and disappears near a = 0.544).

## How to run

```bash
pip install numpy
python koopman_beginner_demo.py
python koopman_dmd_early_warning.py
```

Tested with Python 3.12.3 and NumPy 2.4.4. Random seeds are fixed, so the output is reproducible.

## Cautions

- These are checks on synthetic data with known answers. They do not guarantee the same accuracy on real business data.
- An eigenvalue approaching 1 is a candidate warning sign that a stable state may be weakening; on its own it does not establish that a transition will occur.
- DMD eigenvalues represent the speed of return to a valley bottom only when the data are observed near a stable valley bottom and the local dynamics are approximated there.

## License

CC BY 4.0

## Contact

etalecohomology2026@proton.me
