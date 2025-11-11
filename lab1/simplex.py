from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class OptimizationType(Enum):
    MAXIMIZE = "max"
    MINIMIZE = "min"


class ConstraintType(Enum):
    LEQ = "<="
    GEQ = ">="
    EQ = "="


@dataclass
class Constraint:
    coefficients: np.ndarray
    constraint_type: ConstraintType
    rhs: float


@dataclass
class LinearProgram:
    objective_coeffs: np.ndarray
    constraints: List[Constraint]
    optimization_type: OptimizationType
    unrestricted_indices: List[int] = field(default_factory=list)
    n_original_vars: int = 0

    def __post_init__(self):
        self.n_original_vars = len(self.objective_coeffs)


class SimplexSolver:
    def __init__(self, program: LinearProgram, verbose: bool = True):
        self.lp = program
        self.verbose = verbose
        self.tableau = None
        self.solution = None
        self.optimal_value = None

    def solve(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        if self.verbose:
            print("\n" + "=" * 60)
            print("РЕШЕНИЕ ЗАДАЧИ СИМПЛЕКС-МЕТОДОМ")
            print("=" * 60)

        # Замена unrestricted переменных
        self._handle_unrestricted_variables()

        # Определяем нужен ли двухфазный метод
        needs_artificial = any(
            c.constraint_type in [ConstraintType.EQ, ConstraintType.GEQ]
            for c in self.lp.constraints
        )

        if needs_artificial:
            return self._two_phase_method()
        else:
            return self._standard_simplex()

    def _handle_unrestricted_variables(self):
        """Замена x_i = x_i' - x_i'' для unrestricted переменных"""
        if not self.lp.unrestricted_indices:
            return

        if self.verbose:
            print(f"\nЗамена переменных без ограничения знака:")
            for idx in self.lp.unrestricted_indices:
                print(f"  x{idx + 1} = x{idx + 1}' - x{idx + 1}''")

        # Расширяем objective coefficients
        new_obj = []
        for i, coeff in enumerate(self.lp.objective_coeffs):
            if i in self.lp.unrestricted_indices:
                new_obj.extend([coeff, -coeff])
            else:
                new_obj.append(coeff)

        # Расширяем constraints
        for constraint in self.lp.constraints:
            new_coeffs = []
            for i, coeff in enumerate(constraint.coefficients):
                if i in self.lp.unrestricted_indices:
                    new_coeffs.extend([coeff, -coeff])
                else:
                    new_coeffs.append(coeff)
            constraint.coefficients = np.array(new_coeffs)

        self.lp.objective_coeffs = np.array(new_obj)
        self.lp.unrestricted_indices = []

    def _standard_simplex(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Стандартный симплекс без искусственных переменных"""
        if self.verbose:
            print("\n--- СТАНДАРТНЫЙ СИМПЛЕКС-МЕТОД ---")

        self.tableau = self._build_initial_tableau()
        self._print_tableau("Начальная таблица")

        return self._run_simplex_iterations()

    def _two_phase_method(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Двухфазный симплекс-метод"""
        if self.verbose:
            print("\n--- ДВУХФАЗНЫЙ СИМПЛЕКС-МЕТОД ---")

        # ФАЗА 1
        if self.verbose:
            print("\n>>> ФАЗА 1: Минимизация искусственных переменных")

        self.tableau = self._build_phase1_tableau()
        self._print_tableau("Таблица фазы 1")

        solution, value = self._run_simplex_iterations(phase=1)

        if value is not None and abs(value) > 1e-8:
            if self.verbose:
                print(f"\nЗадача несовместна! Сумма искусственных переменных = {value}")
            return None, None

        # ФАЗА 2
        if self.verbose:
            print("\n>>> ФАЗА 2: Оптимизация исходной функции")

        self.tableau = self._transition_to_phase2()
        self._print_tableau("Таблица фазы 2")

        return self._run_simplex_iterations(phase=2)

    def _build_initial_tableau(self) -> np.ndarray:
        """Построение начальной симплекс-таблицы для стандартного метода"""
        n_vars = len(self.lp.objective_coeffs)
        n_constraints = len(self.lp.constraints)
        n_slack = sum(
            1 for c in self.lp.constraints if c.constraint_type != ConstraintType.EQ
        )

        # Размер: (1 + n_constraints) x (n_vars + n_slack + 1)
        tableau = np.zeros((1 + n_constraints, n_vars + n_slack + 1))

        # Целевая функция (инвертируем для максимизации)
        sign = -1 if self.lp.optimization_type == OptimizationType.MAXIMIZE else 1
        tableau[0, :n_vars] = sign * self.lp.objective_coeffs

        # Ограничения
        slack_col = n_vars
        for i, constraint in enumerate(self.lp.constraints):
            row = i + 1
            tableau[row, :n_vars] = constraint.coefficients
            tableau[row, -1] = constraint.rhs

            if constraint.constraint_type == ConstraintType.LEQ:
                tableau[row, slack_col] = 1
                slack_col += 1
            elif constraint.constraint_type == ConstraintType.GEQ:
                tableau[row, slack_col] = -1
                tableau[row, -1] = -constraint.rhs
                slack_col += 1

        return tableau

    def _build_phase1_tableau(self) -> np.ndarray:
        """Построение таблицы для фазы 1"""
        n_vars = len(self.lp.objective_coeffs)
        n_constraints = len(self.lp.constraints)
        n_slack = sum(
            1 for c in self.lp.constraints if c.constraint_type != ConstraintType.EQ
        )
        n_artificial = sum(
            1
            for c in self.lp.constraints
            if c.constraint_type in [ConstraintType.EQ, ConstraintType.GEQ]
        )

        tableau = np.zeros((1 + n_constraints, n_vars + n_slack + n_artificial + 1))

        # Целевая функция фазы 1: минимизация суммы искусственных
        tableau[0, n_vars + n_slack : n_vars + n_slack + n_artificial] = 1

        # Ограничения
        slack_col = n_vars
        artificial_col = n_vars + n_slack

        for i, constraint in enumerate(self.lp.constraints):
            row = i + 1
            tableau[row, :n_vars] = constraint.coefficients
            tableau[row, -1] = constraint.rhs

            if constraint.constraint_type == ConstraintType.LEQ:
                tableau[row, slack_col] = 1
                slack_col += 1
            elif constraint.constraint_type == ConstraintType.GEQ:
                tableau[row, slack_col] = -1
                tableau[row, artificial_col] = 1
                tableau[row, -1] = -constraint.rhs
                # Корректируем Z-строку
                tableau[0] -= tableau[row]
                slack_col += 1
                artificial_col += 1
            elif constraint.constraint_type == ConstraintType.EQ:
                tableau[row, artificial_col] = 1
                # Корректируем Z-строку
                tableau[0] -= tableau[row]
                artificial_col += 1

        return tableau

    def _transition_to_phase2(self) -> np.ndarray:
        """Переход от фазы 1 к фазе 2"""
        n_vars = len(self.lp.objective_coeffs)
        n_slack = sum(
            1 for c in self.lp.constraints if c.constraint_type != ConstraintType.EQ
        )
        n_constraints = len(self.lp.constraints)

        # Убираем искусственные переменные
        new_tableau = np.zeros((1 + n_constraints, n_vars + n_slack + 1))

        # Копируем тело таблицы (без искусственных переменных)
        new_tableau[1:, : n_vars + n_slack] = self.tableau[1:, : n_vars + n_slack]
        new_tableau[1:, -1] = self.tableau[1:, -1]

        # Устанавливаем исходную целевую функцию
        sign = -1 if self.lp.optimization_type == OptimizationType.MAXIMIZE else 1
        new_tableau[0, :n_vars] = sign * self.lp.objective_coeffs

        # Приводим Z-строку к базисному виду
        for col in range(n_vars + n_slack):
            if self._is_basic_column(new_tableau, col):
                basic_row = self._get_basic_row(new_tableau, col)
                multiplier = new_tableau[0, col]
                new_tableau[0] -= multiplier * new_tableau[basic_row]

        return new_tableau

    def _run_simplex_iterations(
        self, phase: int = 0
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Итерации симплекс-метода"""
        max_iterations = 100

        for iteration in range(max_iterations):
            # Проверка оптимальности
            if self._is_optimal():
                if self.verbose:
                    print(f"\nОптимум найден на итерации {iteration + 1}")
                break

            # Выбор входящей переменной
            entering_col = self._select_entering_variable()
            if entering_col < 0:
                if self.verbose:
                    print("\nНет входящей переменной - задача оптимальна")
                break

            # Выбор выходящей переменной
            leaving_row = self._select_leaving_variable(entering_col)
            if leaving_row < 0:
                if self.verbose:
                    print("\nЗадача неограничена")
                return None, None

            if self.verbose:
                print(
                    f"\nИтерация {iteration + 1}: входит x{entering_col + 1}, выходит строка {leaving_row}"
                )

            # Pivot операция
            self._perform_pivot(leaving_row, entering_col)
            self._print_tableau(f"После итерации {iteration + 1}")

        # Извлечение решения
        self.solution = self._extract_solution()

        # Для максимизации значение берём как есть, для минимизации инвертируем
        if self.lp.optimization_type == OptimizationType.MAXIMIZE:
            self.optimal_value = self.tableau[0, -1]
        else:
            self.optimal_value = -self.tableau[0, -1]

        return self.solution, self.optimal_value

    def _is_optimal(self) -> bool:
        """Проверка условия оптимальности"""
        z_row = self.tableau[0, :-1]
        return np.all(z_row >= -1e-10)

    def _select_entering_variable(self) -> int:
        """Выбор входящей переменной (наименьший отрицательный коэффициент)"""
        z_row = self.tableau[0, :-1]
        min_val = np.min(z_row)

        if min_val >= -1e-10:
            return -1

        return np.argmin(z_row)

    def _select_leaving_variable(self, entering_col: int) -> int:
        """Выбор выходящей переменной (минимальное отношение)"""
        column = self.tableau[1:, entering_col]
        rhs = self.tableau[1:, -1]

        # Вычисляем отношения только для положительных элементов
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(column > 1e-10, rhs / column, np.inf)

        if np.all(ratios == np.inf):
            return -1

        return np.argmin(ratios) + 1  # +1 т.к. пропустили Z-строку

    def _perform_pivot(self, pivot_row: int, pivot_col: int):
        """Выполнение pivot операции"""
        pivot_element = self.tableau[pivot_row, pivot_col]

        # Нормализация pivot строки
        self.tableau[pivot_row] /= pivot_element

        # Обнуление остальных элементов столбца
        for i in range(self.tableau.shape[0]):
            if i != pivot_row:
                multiplier = self.tableau[i, pivot_col]
                self.tableau[i] -= multiplier * self.tableau[pivot_row]

    def _extract_solution(self) -> np.ndarray:
        """Извлечение решения из таблицы"""
        n_vars = len(self.lp.objective_coeffs)
        solution = np.zeros(n_vars)

        for col in range(n_vars):
            if self._is_basic_column(self.tableau, col):
                row = self._get_basic_row(self.tableau, col)
                solution[col] = self.tableau[row, -1]

        return solution

    def _is_basic_column(self, tableau: np.ndarray, col: int) -> bool:
        """Проверка, является ли столбец базисным"""
        column = tableau[1:, col]
        n_ones = np.sum(np.abs(column - 1) < 1e-10)
        n_zeros = np.sum(np.abs(column) < 1e-10)
        return n_ones == 1 and (n_ones + n_zeros) == len(column)

    def _get_basic_row(self, tableau: np.ndarray, col: int) -> int:
        """Получение номера базисной строки для столбца"""
        column = tableau[1:, col]
        return np.argmax(np.abs(column - 1) < 1e-10) + 1

    def _print_tableau(self, title: str = "Таблица"):
        """Вывод симплекс-таблицы"""
        if not self.verbose:
            return

        print(f"\n{title}:")
        print(self.tableau)

    def format_solution(self) -> str:
        """Форматирование решения для вывода"""
        if self.solution is None:
            return "Решение не найдено"

        result = ["Оптимальное решение:"]

        # Обработка unrestricted переменных
        idx = 0
        var_num = 1
        while idx < len(self.solution):
            # Проверяем, была ли переменная unrestricted
            if idx < len(self.solution) - 1 and var_num <= self.lp.n_original_vars:
                # Проверяем паттерн x' и x''
                result.append(f"  x{var_num} = {self.solution[idx]:.6f}")
                idx += 1
                var_num += 1
            else:
                result.append(f"  x{var_num} = {self.solution[idx]:.6f}")
                idx += 1
                var_num += 1

        result.append(f"\nЗначение целевой функции: {self.optimal_value:.6f}")

        return "\n".join(result)


def parse_input_file(filename: str) -> LinearProgram:
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Строка 1: количество переменных и ограничений
    n_vars, n_constraints = map(int, lines[0].split())

    # Строка 2: тип задачи
    opt_type = OptimizationType(lines[1].lower())

    # Строка 3: коэффициенты целевой функции
    obj_coeffs = np.array([float(x) for x in lines[2].split()])

    # Проверка на URS
    current_line = 3
    unrestricted_indices = []
    if current_line < len(lines) and lines[current_line].lower().startswith("urs"):
        urs_vars = list(map(int, lines[current_line].split()[1:]))
        unrestricted_indices = [v - 1 for v in urs_vars]  # 0-based
        current_line += 1
        print(f"Переменные без ограничения знака: {urs_vars}")

    # Ограничения
    constraints = []
    for i in range(current_line, current_line + n_constraints):
        line = lines[i]

        # Определяем тип ограничения
        for sign_str, sign_enum in [
            ("<=", ConstraintType.LEQ),
            (">=", ConstraintType.GEQ),
            ("=", ConstraintType.EQ),
        ]:
            if sign_str in line:
                parts = line.split(sign_str)
                coeffs = np.array([float(x) for x in parts[0].split()])
                rhs = float(parts[1])
                constraints.append(Constraint(coeffs, sign_enum, rhs))
                break

    return LinearProgram(obj_coeffs, constraints, opt_type, unrestricted_indices)


if __name__ == "__main__":
    import sys

    filename = sys.argv[1]

    print(f"Чтение задачи из файла: {filename}")

    # Парсинг
    program = parse_input_file(filename)

    print(f"\nТип задачи: {program.optimization_type.value}")
    print(f"Переменных: {len(program.objective_coeffs)}")
    print(f"Ограничений: {len(program.constraints)}")
    print(f"Целевая функция: {program.objective_coeffs}")

    # Решение
    solver = SimplexSolver(program, verbose=True)
    solution, optimal_value = solver.solve()

    # Вывод результата
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("=" * 60)
    print(solver.format_solution())
