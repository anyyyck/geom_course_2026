# 1 вариант
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def main():
    # 1
    x, a, k = sp.symbols("x a k", real=True)
    f_sym = a * sp.cosh(k * x) ** (-2)
    df_sym = sp.diff(f_sym, x)

    print("Функция профиля волны f(x):", f_sym)
    print("Аналитическая производная df/dx:", df_sym)

    # 2
    a_val = 1.0
    k_val = 0.5
    x_grid = np.linspace(-10.0, 10.0, 500)

    f_num = sp.lambdify(x, f_sym.subs({a: a_val, k: k_val}), modules="numpy")
    df_num = sp.lambdify(x, df_sym.subs({a: a_val, k: k_val}), modules="numpy")

    y_vals = f_num(x_grid)
    dy_vals = df_num(x_grid)

    # 3
    plt.figure(figsize=(9, 5))
    plt.plot(x_grid, y_vals, label=r"Профиль волны $f(x) = a \cdot \cosh^{-2}(kx)$", color="royalblue", lw=2)
    plt.plot(x_grid, dy_vals, label=r"Производная $f'(x)$", color="crimson", lw=2, linestyle="--")

    plt.title(f"Профиль уединённой волны и её производная (a={a_val}, k={k_val})")
    plt.xlabel("Координата x")
    plt.ylabel("Значение функции y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()