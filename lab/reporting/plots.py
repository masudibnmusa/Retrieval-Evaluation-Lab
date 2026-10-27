import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bar_with_ci(lb, primary, path, top_n: int = 15):
    d = lb.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(d) + 1.5))
    err = np.vstack([d[primary] - d["ci_lo"], d["ci_hi"] - d[primary]])
    ax.barh(d["label"], d[primary], xerr=err, capsize=3)
    ax.set_xlabel(primary + " (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def heatmap(lb, primary, path, row="chunker", col="embedder"):
    piv = lb.pivot_table(index=row, columns=col, values=primary, aggfunc="max")
    fig, ax = plt.subplots(figsize=(1.6 * piv.shape[1] + 3, 0.6 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if not np.isnan(piv.values[i, j]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="w")
    fig.colorbar(im, label=f"best {primary}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pareto_plot(lb, primary, path, x="query_usd_per_1k", mask=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lb[x], lb[primary], c="lightgray", label="all")
    if mask is not None:
        ax.scatter(lb.loc[mask, x], lb.loc[mask, primary], c="tab:red", label="Pareto front")
    ax.set_xlabel(x)
    ax.set_ylabel(primary)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)