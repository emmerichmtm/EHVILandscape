import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import norm

# -------------------------------------------------
# 1.  Configuration & training data
# -------------------------------------------------
np.random.seed(42)
GRID          = 200
THETA, Q      = 1.0, 0.5
SIGMA0_SQ     = 1.0
EXPONENT      = 1.2
JITTER        = 1e-9

corner_pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
random_pts = np.random.uniform(0, 1, (60, 2))
X_train    = np.vstack([corner_pts, random_pts])

def true_f(xy):
    return (np.abs(xy[..., 0] - 0.5) ** EXPONENT +
            np.abs(xy[..., 1] - 0.5) ** EXPONENT)

y_train = true_f(X_train)
beta    = y_train.mean()

# -------------------------------------------------
# 2.  GP helpers
# -------------------------------------------------
def corr_matrix(X, theta, q):
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    return np.exp(-theta * D ** q)

def corr_vec(x, X, theta, q):
    return np.exp(-theta * np.linalg.norm(X - x, axis=1) ** q)

def kriging_predict(x, X, y, beta, K_inv, theta, s2, q):
    k = corr_vec(np.asarray(x), X, theta, q)
    w = K_inv @ k
    mu = beta + w @ (y - beta)
    ones = np.ones(len(X))
    sig2 = s2 * (1 - k.T @ w +
                 (1 - ones.T @ w) ** 2 / (ones.T @ K_inv @ ones))
    return mu, np.sqrt(max(sig2, 1e-12))

def expected_improvement(mu, sigma, f_min):
    sigma = np.maximum(sigma, 1e-9)
    z     = (f_min - mu) / sigma
    return (f_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)

# Fit GP
K     = corr_matrix(X_train, THETA, Q)
K    += JITTER * np.eye(len(K))
K_inv = np.linalg.inv(K)
f_min = y_train.min()

# -------------------------------------------------
# 3.  EI grid
# -------------------------------------------------
x1 = np.linspace(0, 1, GRID)
x2 = np.linspace(0, 1, GRID)
X1, X2 = np.meshgrid(x1, x2)

EI_grid = np.zeros((GRID, GRID))
for i in range(GRID):
    for j in range(GRID):
        mu, sigma = kriging_predict((X1[i, j], X2[i, j]),
                                    X_train, y_train,
                                    beta, K_inv,
                                    THETA, SIGMA0_SQ, Q)
        EI_grid[i, j] = expected_improvement(mu, sigma, f_min)

# -------------------------------------------------
# 4.  Local maxima incl. boundary
# -------------------------------------------------
local_mask = np.zeros_like(EI_grid, dtype=bool)
neigh = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]

for i in range(GRID):
    for j in range(GRID):
        neighbors = []
        for di, dj in neigh:
            ni, nj = i + di, j + dj
            if 0 <= ni < GRID and 0 <= nj < GRID:
                neighbors.append(EI_grid[ni, nj])
        if EI_grid[i, j] >= max(neighbors) and any(EI_grid[i, j] > v for v in neighbors):
            local_mask[i, j] = True

local_indices = np.column_stack(np.where(local_mask))
vals          = np.array([EI_grid[i, j] for i, j in local_indices])
order         = np.argsort(vals)[::-1]
local_indices = local_indices[order]
n_peaks       = len(local_indices)
peak_dict     = {tuple(coord): rank for rank, coord in enumerate(map(tuple, local_indices))}

# -------------------------------------------------
# 5.  Greedy ascent basin labels
# -------------------------------------------------
label_grid = -np.ones_like(EI_grid, dtype=int)

def ascend(i, j):
    while not local_mask[i, j]:
        best_i, best_j, best_val = i, j, EI_grid[i, j]
        for di, dj in neigh:
            ni, nj = i + di, j + dj
            if 0 <= ni < GRID and 0 <= nj < GRID and EI_grid[ni, nj] > best_val:
                best_i, best_j, best_val = ni, nj, EI_grid[ni, nj]
        if (best_i, best_j) == (i, j):
            break
        i, j = best_i, best_j
    return peak_dict.get((i, j), -1)

for i in range(GRID):
    for j in range(GRID):
        label_grid[i, j] = ascend(i, j)

# -------------------------------------------------
# 6.  Pastel colormap (exclude white)
# -------------------------------------------------
def pastel_cmap(n, sat=0.35, val=0.95):
    """Return n pastel colours using HSV with fixed low saturation and high value."""
    hues = np.linspace(0, 1, n, endpoint=False)
    rgb  = plt.cm.hsv(hues)[:,:3]
    pastel = sat * rgb + (1 - sat)  # blend toward white with factor (1-sat)
    pastel = val * pastel           # lower value slightly to avoid pure white
    pastel = np.clip(pastel, 0, 1)
    rgba = np.hstack([pastel, np.ones((n,1))])
    return rgba

pastel_colors = pastel_cmap(n_peaks)

# sentinel → light grey
sentinel_rgba = np.array([0.92, 0.92, 0.92, 1.0])
colors = np.vstack([sentinel_rgba, pastel_colors])
plot_grid = label_grid + 1
cmap = ListedColormap(colors)

# -------------------------------------------------
# 7.  Plot
# -------------------------------------------------
plt.figure(figsize=(6.5, 6))
plt.imshow(plot_grid, origin="lower", extent=[0, 1, 0, 1],
           cmap=cmap, interpolation="nearest")
plt.contour(X1, X2, EI_grid, levels=12, colors="k",
            linewidths=0.35, alpha=0.5)

for idx, (pi, pj) in enumerate(local_indices):
    xp, yp = X1[pi,pj], X2[pi,pj]
    plt.scatter(xp, yp, c=[pastel_colors[idx]], edgecolor="black",
                marker="*", s=150, zorder=3)
    plt.text(xp + 0.012, yp + 0.012, f"P{idx}", fontsize=7, weight="bold")

plt.title(" ")
plt.xlabel("$x_1$"); plt.ylabel("$x_2$")
plt.tight_layout()
plt.show()

print(f"Detected local maxima (height‑ordered): {n_peaks}")
