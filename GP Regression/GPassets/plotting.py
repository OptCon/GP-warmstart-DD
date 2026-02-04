import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_errors(train, test, kernel_name, fname,
                figsize=(3.9, 3.8),
                train_color="tab:orange",
                test_color="tab:blue",
                line_width=0.8,
                annotate=False,          
                pad_decades=0.25,       
                floor=1e-12):            
    """
    Save a 2x1 (train/test) error plot sized for a single-column paper figure.
    Uses log y-scale with data-driven y-limits (autoscaled per subplot).
    """

    train = np.asarray(train).ravel()
    test  = np.asarray(test).ravel()
    x_tr = np.arange(train.size)
    x_te = np.arange(test.size)

    fig, ax = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)

    ax[0].plot(x_tr, train, color=train_color, lw=line_width)
    ax[1].plot(x_te, test,  color=test_color,  lw=line_width)

    ax[0].set_ylabel(r'$\|\hat{\lambda}-\lambda^\star\|_{\infty}$')
    ax[1].set_ylabel(r'$\|\hat{\lambda}-\lambda^\star\|_{\infty}$')
    ax[1].set_xlabel("Sample #")

    def autoscale_y_log(a, y):
        y = np.asarray(y)
        y = y[np.isfinite(y)]
        y = y[y > 0]
        if y.size == 0:
            return

        ymin, ymax = float(y.min()), float(y.max())

        lo = np.log10(max(ymin, floor)) - pad_decades
        hi = np.log10(max(ymax, floor)) + pad_decades

        a.set_yscale("log")
        a.set_ylim(10**lo, 10**hi)

    autoscale_y_log(ax[0], train)
    autoscale_y_log(ax[1], test)

    for a in ax:
        a.grid(True, which="both", alpha=0.3)

    if annotate:
        def mark_min_max(a, x, y):
            if y.size == 0:
                return
            y_pos = np.asarray(y)
            mask = np.isfinite(y_pos) & (y_pos > 0)
            if not np.any(mask):
                return

            x2 = x[mask]
            y2 = y_pos[mask]

            i_min = int(np.argmin(y2))
            i_max = int(np.argmax(y2))
            idx = [i_min] if i_min == i_max else [i_min, i_max]

            a.scatter(x2[idx], y2[idx],
                      s=18, marker="x", c="black",
                      linewidths=1.0, zorder=3)

            xmin, xmax = float(x2.min()), float(x2.max())
            ymin, ymax = float(y2.min()), float(y2.max())

            for i in idx:
                xr = 0.0 if xmax == xmin else (float(x2[i]) - xmin) / (xmax - xmin)
                yr = 0.0 if ymax == ymin else (float(y2[i]) - ymin) / (ymax - ymin)

                dx = -28 if xr > 0.85 else 6
                dy = -14 if yr > 0.85 else 6
                ha = "right" if dx < 0 else "left"
                va = "top"   if dy < 0 else "bottom"

                a.annotate(f"{y2[i]:.2e}",
                           xy=(x2[i], y2[i]),
                           xytext=(dx, dy),
                           textcoords="offset points",
                           fontsize=8,
                           ha=ha, va=va,
                           bbox=dict(boxstyle="round,pad=0.2",
                                     fc="white", ec="none", alpha=0.85),
                           clip_on=True)

        mark_min_max(ax[0], x_tr, train)
        mark_min_max(ax[1], x_te, test)
    plt.show()
    fig.savefig(fname, bbox_inches="tight")

def plot_iters(ar1, ar2, ar3,
                labels=("ReLU", "Matern", "Cold-Start"),
                figsize=(3.48, 3.6),
                line_width=1.5,
                annotate=False,
                filename="plot.pdf"):

    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()
    ar3 = np.asarray(ar3).ravel()

    n = min(ar1.size, ar2.size, ar3.size)
    if n == 0:
        raise ValueError("Empty input arrays.")
    if ar1.size != ar2.size:
        ar1 = ar1[:n]
        ar2 = ar2[:n]
        ar3 = ar2[:n]

    x = np.arange(n)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.plot(x, ar1, color="tab:red", lw=line_width, label=labels[0])
    ax.plot(x, ar2, color="tab:orange",   lw=line_width, label=labels[1])
    ax.plot(x, ar3, color="tab:blue",   lw=line_width, label=labels[2])

    ax.set_ylabel("Dual decomposition iterations")
    ax.set_xlabel("Sample #")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8,
              loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0.)
    
    def place_dx(idx):
        xmin, xmax = float(x.min()), float(x.max())
        xr = 0.0 if xmax == xmin else (x[idx] - xmin) / (xmax - xmin)
        return (-36, "right") if xr > 0.80 else (8, "left")

    if annotate:
        # Worst case in cold-start
        i = int(np.argmax(ar2))
        y_cold = float(ar2[i])
        y_warm = float(ar1[i])

        ax.scatter([x[i]], [y_cold], s=40, marker="x", c="black", linewidths=1.2, zorder=4)
        ax.scatter([x[i]], [y_warm], s=40, marker="x", c="black", linewidths=1.2, zorder=4)

        dx, ha = place_dx(i)
        ax.annotate(f"{y_cold:.0f}", xy=(x[i], y_cold), xytext=(dx, -5),
                    textcoords="offset points", ha=ha, va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.6),
                    clip_on=True)

        ax.annotate(f"{y_warm:.0f}", xy=(x[i], y_warm), xytext=(dx, 5),
                    textcoords="offset points", ha=ha, va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.6),
                    clip_on=True)

        # Best improvement (only if positive)
        diff = ar2 - ar1
        j = int(np.argmax(diff))
        best_impr = float(diff[j])

        if best_impr > 0:
            y_cold_best = float(ar2[j])
            y_warm_best = float(ar1[j])

            ax.scatter([x[j]], [y_cold_best], s=40, marker="^", c="black", zorder=4)
            ax.scatter([x[j]], [y_warm_best], s=40, marker="^", c="black", zorder=4)

            dx2, ha2 = place_dx(j)
            ax.annotate(rf"${y_cold_best:.0f}$", xy=(x[j], y_cold_best), xytext=(dx2, -5),
                        textcoords="offset points", ha=ha2, va="bottom", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                        arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.6),
                        clip_on=True)
            ax.annotate(rf"${y_warm_best:.0f}$", xy=(x[j], y_warm_best), xytext=(dx2, 5),
                    textcoords="offset points", ha=ha2, va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.6),
                    clip_on=True)

    # Save first, then show
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)


def plot_arrays(ar1, ar2, ar3, 
                labels=("ReLU", "Matern", "Cold-Start"),
                figsize=(4.3, 2.8),   # IFAC single-column width
                line_width=1.5,
                filename="plot.pdf"):


    # Manually add axes so we can reserve space on right for legend
    # [left, bottom, width, height] in figure coords
   

    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()
    ar3 = np.asarray(ar3).ravel()

    n = min(ar1.size, ar2.size, ar3.size)
    if n == 0:
        raise ValueError("Empty input arrays.")
    if ar1.size != ar2.size:
        ar1 = ar1[:n]
        ar2 = ar2[:n]
        ar3 = ar2[:n]

    x = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(x, ar1, color="tab:red", lw=line_width, label=labels[0])
    ax.plot(x, ar2, color="tab:orange",   lw=line_width, label=labels[1])
    ax.plot(x, ar3, color="tab:blue",   lw=line_width, label=labels[2]) 
    ax.set_ylabel("DD iterations")
    ax.set_xlabel("Sample #")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))


    # Create the legend in a *separate* axes area
    leg_ax = fig.add_axes([0.7, 0.13, 0.25, 0.80])  # right area [0.75, 0.03, 0.25, 0.80]  [0.7, 0.13, 0.25, 0.80]
    leg = leg_ax.legend(
        ax.lines,         # use all lines from ax
        labels,
        frameon=False,
        fontsize=8,
        loc="center"
    )

    leg_ax.axis("off")  # hide the legend axes frame

    # Save and show
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)