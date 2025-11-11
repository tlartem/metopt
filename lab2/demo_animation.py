from animation import OptimizationAnimator
from global_optimizer import GlobalOptimizer


def demo_animation():
    # Раскомментируйте нужную функцию для запуска

    # Простая функция с несколькими минимумами
    # func = "x + sin(3.14159*x)"
    # a, b = -2.0, 2.0

    # Функция Растригина
    # func = "x**2 - 10*cos(2*pi*x) + 10"
    # a, b = -5.0, 5.0

    # Функция Экли
    # func = "-20*exp(-0.2*sqrt(x**2)) - exp(cos(2*pi*x)) + 20 + e"
    # a, b = -5.0, 5.0

    # Функция с квадратом и синусом
    func = "x**2 + sin(5*x)"
    a, b = -3.0, 3.0

    eps = 0.01
    optimizer = GlobalOptimizer(func, a, b, eps)
    animator = OptimizationAnimator(optimizer, interval=300, save_path=None)
    animator.animate_optimization()


def demo_animation_save():
    func = "x + sin(3.14159*x)"
    a, b = -2.0, 2.0
    eps = 0.01
    optimizer = GlobalOptimizer(func, a, b, eps)
    animator = OptimizationAnimator(
        optimizer, interval=200, save_path="optimization_animation.gif"
    )
    animator.animate_optimization()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "save":
        demo_animation_save()
    else:
        demo_animation()
