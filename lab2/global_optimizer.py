import time
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class GlobalOptimizer:
    def __init__(self, func_str: str, a: float, b: float, eps: float):
        """
        Args:
            func_str: строка с функцией (например, "x + sin(3.14159*x)")
            a: левая граница отрезка
            b: правая граница отрезка
            eps: требуемая точность
        """
        self.func_str = func_str
        self.func = self._parse_function(func_str)
        self.a = a
        self.b = b
        self.eps = eps

        # Оценка липшицевой константы через максимум производной
        self.L = self._estimate_lipschitz_constant()

        # История вычислений
        self.evaluated_points: List[Tuple[float, float]] = []  # (x, f(x))
        self.iterations = 0
        self.start_time = None

    def _parse_function(self, func_str: str) -> Callable:
        """Парсинг функции из строки"""
        # Заменяем математические функции на numpy эквиваленты
        func_str = func_str.replace("sin", "np.sin")
        func_str = func_str.replace("cos", "np.cos")
        func_str = func_str.replace("exp", "np.exp")
        func_str = func_str.replace("log", "np.log")
        func_str = func_str.replace("sqrt", "np.sqrt")
        func_str = func_str.replace("abs", "np.abs")
        func_str = func_str.replace("pi", "np.pi")
        func_str = func_str.replace("e", "np.e")

        # Создаем функцию
        def f(x):
            try:
                return eval(func_str, {"np": np, "x": x})
            except:
                return np.nan

        return f

    def _estimate_lipschitz_constant(self, n_samples: int = 1000) -> float:
        """Оценка липшицевой константы через численное дифференцирование"""
        x_samples = np.linspace(self.a, self.b, n_samples)
        h = (self.b - self.a) / n_samples

        # Вычисляем значения функции, пропуская nan
        y_samples = []
        for x in x_samples:
            val = self.func(x)
            if not np.isnan(val) and not np.isinf(val):
                y_samples.append(val)
            else:
                y_samples.append(0.0)

        # Численная производная
        df = np.abs(np.gradient(y_samples, h))

        # Убираем nan и inf
        df = df[~np.isnan(df)]
        df = df[~np.isinf(df)]

        if len(df) == 0:
            return 1.0

        # Берем максимум с запасом
        L = np.max(df) * 1.1
        return max(L, 1.0)  # Минимум 1.0

    def _lower_bound(self, x: float, x_i: float, f_i: float) -> float:
        """Нижняя оценка функции в точке x на основе значения в точке x_i"""
        return -self.L * np.abs(x - x_i) + f_i

    def _find_intersection(self, x1: float, f1: float, x2: float, f2: float) -> float:
        """Нахождение точки пересечения двух вспомогательных функций"""
        # Уравнение: -L|x - x1| + f1 = -L|x - x2| + f2
        # Рассматриваем случаи в зависимости от расположения точек

        if x1 == x2:
            return x1

        # Случай 1: x между x1 и x2
        # -L(x - x1) + f1 = L(x - x2) + f2
        # -Lx + Lx1 + f1 = Lx - Lx2 + f2
        # -2Lx = -Lx1 - f1 - Lx2 + f2
        # x = (Lx1 + f1 + Lx2 - f2) / (2L)

        x_intersect = (self.L * x1 + f1 + self.L * x2 - f2) / (2 * self.L)

        # Проверяем, что точка в допустимых пределах
        if x_intersect < self.a:
            return self.a
        if x_intersect > self.b:
            return self.b

        return x_intersect

    def _find_next_point(self) -> float:
        """Нахождение следующей точки для вычисления функции"""
        if len(self.evaluated_points) == 0:
            return (self.a + self.b) / 2

        if len(self.evaluated_points) == 1:
            x1, f1 = self.evaluated_points[0]
            # Проверяем границы
            if abs(x1 - self.a) > abs(x1 - self.b):
                return self.a
            else:
                return self.b

        # Сортируем точки по x
        sorted_points = sorted(self.evaluated_points, key=lambda p: p[0])

        # Находим точку с минимальной нижней оценкой среди всех пересечений
        min_lower = float("inf")
        best_x = None

        # Проверяем границы
        for boundary in [self.a, self.b]:
            # Находим ближайшую точку
            closest = min(sorted_points, key=lambda p: abs(p[0] - boundary))
            lower = self._lower_bound(boundary, closest[0], closest[1])
            if lower < min_lower:
                min_lower = lower
                best_x = boundary

        # Проверяем пересечения между соседними точками
        for i in range(len(sorted_points) - 1):
            x1, f1 = sorted_points[i]
            x2, f2 = sorted_points[i + 1]

            x_intersect = self._find_intersection(x1, f1, x2, f2)

            # Проверяем, что точка между x1 и x2
            if x1 < x_intersect < x2:
                lower = self._lower_bound(x_intersect, x1, f1)
                if lower < min_lower:
                    min_lower = lower
                    best_x = x_intersect

        return best_x if best_x is not None else (self.a + self.b) / 2

    def _check_convergence(self) -> bool:
        """Проверка условия остановки"""
        if len(self.evaluated_points) < 2:
            return False

        # Находим минимальное значение функции
        min_f = min(f for _, f in self.evaluated_points)

        # Находим минимальную нижнюю оценку
        sorted_points = sorted(self.evaluated_points, key=lambda p: p[0])
        min_lower = float("inf")

        # Проверяем границы
        for boundary in [self.a, self.b]:
            closest = min(sorted_points, key=lambda p: abs(p[0] - boundary))
            lower = self._lower_bound(boundary, closest[0], closest[1])
            min_lower = min(min_lower, lower)

        # Проверяем пересечения
        for i in range(len(sorted_points) - 1):
            x1, f1 = sorted_points[i]
            x2, f2 = sorted_points[i + 1]
            x_intersect = self._find_intersection(x1, f1, x2, f2)
            if x1 < x_intersect < x2:
                lower = self._lower_bound(x_intersect, x1, f1)
                min_lower = min(min_lower, lower)

        # Проверка: разница между минимальным значением и минимальной нижней оценкой
        return (min_f - min_lower) <= self.eps

    def optimize(self) -> Tuple[float, float, int, float]:
        """
        Выполнение оптимизации

        Returns:
            (x_min, f_min, iterations, elapsed_time)
        """
        self.start_time = time.time()
        self.iterations = 0
        self.evaluated_points = []

        max_iterations = 10000

        # Начальные точки на границах
        f_a = self.func(self.a)
        f_b = self.func(self.b)
        self.evaluated_points.append((self.a, f_a))
        self.evaluated_points.append((self.b, f_b))
        self.iterations += 2

        while not self._check_convergence() and self.iterations < max_iterations:
            next_x = self._find_next_point()

            # Избегаем повторных вычислений в одной точке
            if any(abs(x - next_x) < 1e-10 for x, _ in self.evaluated_points):
                break

            next_f = self.func(next_x)
            self.evaluated_points.append((next_x, next_f))
            self.iterations += 1

        elapsed_time = time.time() - self.start_time

        # Находим минимум
        min_idx = np.argmin([f for _, f in self.evaluated_points])
        x_min, f_min = self.evaluated_points[min_idx]

        return x_min, f_min, self.iterations, elapsed_time

    def visualize(self, save_path: str = None):
        """Визуализация функции и процесса оптимизации"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # График исходной функции
        x_plot = np.linspace(self.a, self.b, 1000)
        y_plot = [self.func(x) for x in x_plot]
        ax.plot(x_plot, y_plot, "b-", linewidth=2, label="f(x)")

        # Вспомогательные функции (ломаная линия)
        if len(self.evaluated_points) > 1:
            sorted_points = sorted(self.evaluated_points, key=lambda p: p[0])

            # Строим нижние оценки - ломаную линию
            x_lower = []
            y_lower = []

            # Для каждого сегмента между соседними точками
            for i in range(len(sorted_points) - 1):
                x1, f1 = sorted_points[i]
                x2, f2 = sorted_points[i + 1]

                # Находим точку пересечения
                x_intersect = self._find_intersection(x1, f1, x2, f2)

                # Строим две части: от x1 до пересечения и от пересечения до x2
                if x1 < x_intersect < x2:
                    # От x1 до пересечения используем оценку от x1
                    x_part1 = np.linspace(x1, x_intersect, 50)
                    for x in x_part1:
                        y = self._lower_bound(x, x1, f1)
                        x_lower.append(x)
                        y_lower.append(y)

                    # От пересечения до x2 используем оценку от x2
                    x_part2 = np.linspace(x_intersect, x2, 50)
                    for x in x_part2:
                        y = self._lower_bound(x, x2, f2)
                        x_lower.append(x)
                        y_lower.append(y)
                else:
                    # Если пересечение вне сегмента, используем обе оценки и берем максимум
                    x_segment = np.linspace(x1, x2, 100)
                    for x in x_segment:
                        y1 = self._lower_bound(x, x1, f1)
                        y2 = self._lower_bound(x, x2, f2)
                        y = max(y1, y2)  # Максимум нижних оценок
                        x_lower.append(x)
                        y_lower.append(y)

            ax.plot(
                x_lower, y_lower, "r--", linewidth=1.5, alpha=0.7, label="Нижняя оценка"
            )

        # Отмеченные точки
        x_vals = [x for x, _ in self.evaluated_points]
        y_vals = [f for _, f in self.evaluated_points]
        ax.scatter(x_vals, y_vals, c="red", s=50, zorder=5, label="Вычисленные точки")

        # Минимум
        min_idx = np.argmin(y_vals)
        x_min, f_min = self.evaluated_points[min_idx]
        ax.scatter(
            [x_min],
            [f_min],
            c="green",
            s=200,
            marker="*",
            zorder=6,
            label=f"Найденный минимум: x={x_min:.6f}, f={f_min:.6f}",
        )

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("f(x)", fontsize=12)
        ax.set_title(f"Поиск глобального минимума: {self.func_str}", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
