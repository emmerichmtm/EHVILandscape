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
SIGMA0_SQ     = 1.0     # process variance
EXPONENT      = 1.2
JITTER        = 1e-9    # nugget for numerical stability

# Training points: 4 corners + 60 random samples
corner_pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
random_pts = np.random.uniform(0, 1, (60, 2))
X_train    = np.vstack([corner_pts, random_pts])

def true_f(xy):
    return (np.abs(xy[..., 0] - 0.5) ** EXPONENT +
            np.abs(xy[..., 1] - 0.5) ** EXPONENT)

y_train = true_f(X_train)
beta    = y_train.mean()

# -------------------------------------------------
# 2.  Gaussian‑process utilities
# -------------------------------------------------
def corr_matrix(X, theta, q):
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    return np.exp(-theta * D ** q)

def corr_vec(x, X, theta, q):
    return np.exp(-theta * np.linalg.norm(X - x, axis=1) ** q)

def kriging_predict(x, X, y, beta, K_inv, theta, s2, q):
    k     = corr_vec(np.asarray(x), X, theta, q)
    w     = K_inv @ k
    mu    = beta + w @ (y - beta)
    ones  = np.ones(len(X))
    sig2  = s2 * (1 - k.T @ w +
                  (1 - ones.T @ w) ** 2 / (ones.T @ K_inv @ ones))
    sigma = np.sqrt(max(sig2, 1e-12))            # clip tiny negatives
    return mu, sigma

def expected_improvement(mu, sigma, f_min):
    sigma = np.maximum(sigma, 1e-9)              # avoid divide‑by‑zero
    z     = (f_min - mu) / sigma
    return (f_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)

# Fit GP
K     = corr_matrix(X_train, THETA, Q)
K    += JITTER * np.eye(len(K))
K_inv = np.linalg.inv(K)
f_min = y_train.min()

# -------------------------------------------------
# 3.  Evaluate EI on a dense grid
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
# 4.  Identify strict local maxima (including boundaries)
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
        if neighbors:
            if EI_grid[i, j] >= max(neighbors) and any(EI_grid[i, j] > v for v in neighbors):
                local_mask[i, j] = True

# Sort peaks by EI value (descending)
local_indices = np.column_stack(np.where(local_mask))
peak_vals     = np.array([EI_grid[i, j] for i, j in local_indices])
order         = np.argsort(peak_vals)[::-1]
local_indices = local_indices[order]
n_peaks       = len(local_indices)

# Map peak location to rank
peak_dict = {tuple(coord): idx for idx, coord in enumerate(map(tuple, local_indices))}

# -------------------------------------------------
# 5.  Greedy ascent basins
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
# 6.  Pastel colormap helper
# -------------------------------------------------
def pastel_cmap(n):
    # base HSV hues
    hues = np.linspace(0, 1, n, endpoint=False)
    base = plt.cm.hsv(hues)
    # blend with white to make pastel
    pastel = 0.6 * base[:, :3] + 0.4  # rgb components
    pastel = np.hstack([pastel, np.ones((n,1))])  # add alpha
    return pastel

# Build final ListedColormap
plot_grid = label_grid + 1  # sentinel -1 -> 0
pastel_colors = pastel_cmap(n_peaks)
colors = np.vstack([np.array([1,1,1,0]), pastel_colors])  # prepend transparent
cmap = ListedColormap(colors)

# -------------------------------------------------
# 7.  Plotting
# -------------------------------------------------
plt.figure(figsize=(6.5, 6))
plt.imshow(plot_grid, origin="lower", extent=[0, 1, 0, 1],
           cmap=cmap, interpolation="nearest", alpha=0.9)
plt.contour(X1, X2, EI_grid, levels=12, colors="k",
            linewidths=0.4, alpha=0.5)

# Mark peaks
for idx, (pi, pj) in enumerate(local_indices):
    x_peak, y_peak = X1[pi, pj], X2[pi, pj]
    plt.scatter(x_peak, y_peak, c=[pastel_colors[idx]], edgecolor="black",
                marker="*", s=150, zorder=3)
    plt.text(x_peak + 0.01, y_peak + 0.01, f"P{idx}", fontsize=7, weight="bold")

plt.title("Pastel Basins of Attraction of EI Peaks", fontsize=13)
plt.xlabel("$x_1$"); plt.ylabel("$x_2$")
plt.tight_layout()
plt.show()

print(f"Detected local maxima (height-ordered): {n_peaks}")
