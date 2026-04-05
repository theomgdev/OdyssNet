"""
Training history tracker and plotting utility for OdyssNet examples.

Usage:
    from odyssnet.utils.history import TrainingHistory

    history = TrainingHistory()

    for epoch in range(epochs):
        loss = trainer.train_batch(...)
        history.record(loss=loss, lr=1e-3, accuracy=acc)

    history.plot()                       # show interactive plot
    history.plot(save_path="plot.png")   # save to file
"""

import os


class TrainingHistory:
    """Lightweight metric accumulator with built-in plotting."""

    def __init__(self):
        self._data: dict[str, list[float]] = {}

    def record(self, **kwargs: float):
        """Record one or more named metrics for the current step.

        Example::

            history.record(loss=0.5, lr=1e-3, accuracy=0.92)
        """
        for key, value in kwargs.items():
            if key not in self._data:
                self._data[key] = []
            self._data[key].append(float(value))

    def get(self, key: str) -> list[float]:
        """Return recorded values for a metric."""
        return self._data.get(key, [])

    @property
    def metrics(self) -> list[str]:
        """Return names of all recorded metrics."""
        return list(self._data.keys())

    def plot(self, save_path: str | None = None, title: str = "Training History"):
        """Plot all recorded metrics.

        Each metric gets its own subplot with a shared x-axis (step).
        If ``save_path`` is provided the figure is saved to disk instead of
        being displayed interactively.
        """
        try:
            import matplotlib
            if save_path:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[TrainingHistory] matplotlib not installed — skipping plot.")
            return

        keys = [k for k in self._data if len(self._data[k]) > 0]
        if not keys:
            print("[TrainingHistory] No metrics recorded — nothing to plot.")
            return

        n = len(keys)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True, squeeze=False)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        for i, key in enumerate(keys):
            ax = axes[i][0]
            values = self._data[key]
            ax.plot(values, linewidth=1.2)
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)

            # Annotate min/max for loss-like metrics
            if len(values) > 1:
                best_idx = min(range(len(values)), key=lambda j: values[j])
                ax.annotate(
                    f"min={values[best_idx]:.4g}",
                    xy=(best_idx, values[best_idx]),
                    fontsize=8,
                    color="green",
                    ha="center",
                    va="top",
                )

        axes[-1][0].set_xlabel("Step")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[TrainingHistory] Plot saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()
