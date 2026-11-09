import numpy as np

def bisection(f, a: float, b: float, eps: float = 1e-6) -> float:
    """Solutions f(x) = 0 """
    if f(a) * f(b) >= 0:
        raise ValueError(
            "Required values f(a) * f(b) < 0"
        )

    while (b - a) >= eps:
        c = (a + b) / 2.0
        fc = f(c)
        if fc == 0 or abs(fc) < eps:
            return c
        if f(a) * fc < 0:
            b = c
        else:
            a = c

    return (a + b) / 2.0

def main():
    def func(x):
        return x**3 - x - 2

    a, b = 1.0, 2.0
    eps = 1e-6

    root = bisection(func, a, b, eps)

    print(f"Equation: x^3 - x - 2 = 0 on [{a}, {b}]")
    print(f"Accuracy: {eps}")
    print(f"Root: {root:.6f}")
    print(f"Examination f(x): {func(root):.2e}")

if __name__ == "__main__":
    main()