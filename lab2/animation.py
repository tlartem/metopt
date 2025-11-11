import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class OptimizationAnimator:
    def __init__(self, optimizer, interval: int = 500, save_path: str = None):
        self.optimizer = optimizer
        self.interval = interval
        self.save_path = save_path
        self.history: List[Tuple[float, float]] = []
        self.current_iteration = 0

    def _setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))

        x_plot = np.linspace(self.optimizer.a, self.optimizer.b, 1000)
        y_plot = [self.optimizer.func(x) for x in x_plot]
        (self.func_line,) = self.ax.plot(
            x_plot, y_plot, "b-", linewidth=2, label="f(x)"
        )

        (self.lower_line,) = self.ax.plot(
            [], [], "r--", linewidth=1.5, alpha=0.7, label="Нижняя оценка"
        )

        self.points_scatter = self.ax.scatter(
            [], [], c="red", s=50, zorder=5, label="Точки"
        )

        self.min_scatter = self.ax.scatter(
            [], [], c="green", s=200, marker="*", zorder=6, label="Минимум"
        )

        self.ax.set_xlabel("x", fontsize=12)
        self.ax.set_ylabel("f(x)", fontsize=12)
        self.ax.set_title(f"{self.optimizer.func_str}", fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(fontsize=10)

    def _update_animation(self, frame):
        if frame >= len(self.history):
            return

        current_points = self.history[: frame + 1]
        if len(current_points) == 0:
            return

        x_vals = [x for x, _ in current_points]
        y_vals = [f for _, f in current_points]
        self.points_scatter.set_offsets(list(zip(x_vals, y_vals)))

        min_idx = np.argmin(y_vals)
        x_min, f_min = current_points[min_idx]
        self.min_scatter.set_offsets([[x_min, f_min]])

        if len(current_points) > 1:
            sorted_points = sorted(current_points, key=lambda p: p[0])
            x_lower, y_lower = self._build_lower_bound_line(sorted_points)
            self.lower_line.set_data(x_lower, y_lower)

        return (
            self.func_line,
            self.lower_line,
            self.points_scatter,
            self.min_scatter,
        )

    def _build_lower_bound_line(self, sorted_points):
        x_lower = []
        y_lower = []

        for i in range(len(sorted_points) - 1):
            x1, f1 = sorted_points[i]
            x2, f2 = sorted_points[i + 1]
            x_intersect = self.optimizer._find_intersection(x1, f1, x2, f2)

            if x1 < x_intersect < x2:
                x_part1 = np.linspace(x1, x_intersect, 50)
                for x in x_part1:
                    y = self.optimizer._lower_bound(x, x1, f1)
                    x_lower.append(x)
                    y_lower.append(y)

                x_part2 = np.linspace(x_intersect, x2, 50)
                for x in x_part2:
                    y = self.optimizer._lower_bound(x, x2, f2)
                    x_lower.append(x)
                    y_lower.append(y)
            else:
                x_segment = np.linspace(x1, x2, 100)
                for x in x_segment:
                    y1 = self.optimizer._lower_bound(x, x1, f1)
                    y2 = self.optimizer._lower_bound(x, x2, f2)
                    y = max(y1, y2)
                    x_lower.append(x)
                    y_lower.append(y)

        return x_lower, y_lower

    def animate_optimization(self, callback=None):
        self._setup_plot()

        if callback:
            callback()
        else:
            self._run_optimization_with_callback()

        anim = FuncAnimation(
            self.fig,
            self._update_animation,
            frames=len(self.history),
            interval=self.interval,
            blit=True,
            repeat=True,
        )

        if self.save_path:
            anim.save(self.save_path, writer="pillow", fps=1000 / self.interval)
        else:
            plt.show()

        return anim

    def add_point(self, x: float, f: float):
        self.history.append((x, f))

    def _run_optimization_with_callback(self):
        self.optimizer.start_time = time.time()
        self.optimizer.iterations = 0
        self.optimizer.evaluated_points = []

        max_iterations = 10000

        # Начальные точки на границах
        f_a = self.optimizer.func(self.optimizer.a)
        f_b = self.optimizer.func(self.optimizer.b)
        self.optimizer.evaluated_points.append((self.optimizer.a, f_a))
        self.optimizer.evaluated_points.append((self.optimizer.b, f_b))
        self.add_point(self.optimizer.a, f_a)
        self.add_point(self.optimizer.b, f_b)
        self.optimizer.iterations += 2

        while (
            not self.optimizer._check_convergence()
            and self.optimizer.iterations < max_iterations
        ):
            next_x = self.optimizer._find_next_point()

            if any(abs(x - next_x) < 1e-10 for x, _ in self.optimizer.evaluated_points):
                break

            next_f = self.optimizer.func(next_x)
            self.optimizer.evaluated_points.append((next_x, next_f))
            self.add_point(next_x, next_f)
            self.optimizer.iterations += 1


def create_animated_optimization(
    func_str: str,
    a: float,
    b: float,
    eps: float,
    interval: int = 500,
    save_path: str = None,
):
    from global_optimizer import GlobalOptimizer

    optimizer = GlobalOptimizer(func_str, a, b, eps)
    animator = OptimizationAnimator(optimizer, interval=interval, save_path=save_path)
    anim = animator.animate_optimization()
    return anim, optimizer
