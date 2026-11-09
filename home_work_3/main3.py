import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

def rhs(t: float, y: np.ndarray, L1: float, L2: float, m1: float, m2: float, g: float) -> np.ndarray:
    """Right-hand side of the four first-order ODEs (eq. 5.5)."""
    theta1, omega1, theta2, omega2 = y
    delta = theta1 - theta2
    denom = 2 * m1 + m2 - m2 * np.cos(2 * delta)

    dtheta1 = omega1
    domega1 = (
        -g * (2 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2 * theta2)
        - 2 * np.sin(delta) * m2 * (omega2**2 * L2 + omega1**2 * L1 * np.cos(delta))
    ) / (L1 * denom)

    dtheta2 = omega2
    domega2 = (
        2 * np.sin(delta) * (
            omega1**2 * L1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta1)
            + omega2**2 * L2 * m2 * np.cos(delta)
        )
    ) / (L2 * denom)

    return np.array([dtheta1, domega1, dtheta2, domega2], dtype=float)

def get_cartesian(theta1: np.ndarray, theta2: np.ndarray, L1: float, L2: float):
    """Convert angles to Cartesian coordinates of the bobs (eq. 5.1)."""
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2

def total_energy(y: np.ndarray, L1: float, L2: float, m1: float, m2: float, g: float) -> np.ndarray:
    """Total mechanical energy E = T + V (eq. 5.2 and 5.3)."""
    theta1, omega1, theta2, omega2 = y
    T = (
        0.5 * (m1 + m2) * (L1**2) * (omega1**2)
        + 0.5 * m2 * (L2**2) * (omega2**2)
        + m2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2)
    )
    V = -(m1 + m2) * g * L1 * np.cos(theta1) - m2 * g * L2 * np.cos(theta2)
    return T + V

def main():
    # Parameters from Table 1
    m1, m2 = 1.0, 1.0
    L1, L2 = 1.0, 1.0
    g = 9.81
    t_span = (0.0, 20.0)
    N = 1000
    t_eval = np.linspace(t_span[0], t_span[1], N)

    # Initial conditions
    y0_A = np.array([np.pi / 2, 0.0, np.pi / 2 + 0.01, 0.0])
    delta_0 = 1e-6
    y0_B = np.array([np.pi / 2, 0.0, np.pi / 2 + 0.01 + delta_0, 0.0])

    # --- TASK 1: Numerical solution (Case A, tolerance 1e-8) ---
    sol_A = solve_ivp(
        rhs, t_span, y0_A, method="RK45", t_eval=t_eval,
        args=(L1, L2, m1, m2, g), rtol=1e-8, atol=1e-8
    )
    x1_A, y1_A, x2_A, y2_A = get_cartesian(sol_A.y[0], sol_A.y[2], L1, L2)

    # 1.4. Static trajectory plot of the lower bob
    plt.figure(figsize=(6, 6))
    plt.plot(x2_A, y2_A, lw=1.2, color="royalblue")
    plt.title("Lower bob trajectory $(x_2(t), y_2(t))$")
    plt.xlabel("$x$, m")
    plt.ylabel("$y$, m")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    # --- TASK 2: Instability check and energy conservation ---
    sol_B = solve_ivp(
        rhs, t_span, y0_B, method="RK45", t_eval=t_eval,
        args=(L1, L2, m1, m2, g), rtol=1e-8, atol=1e-8
    )
    _, _, x2_B, y2_B = get_cartesian(sol_B.y[0], sol_B.y[2], L1, L2)

    # 2.2. Side-by-side comparison (Cases A and B)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].plot(x2_A, y2_A, lw=1.1, color="royalblue")
    axes[0].set_title(r"Case A: $\theta_2(0) = \pi/2 + 0.01$")
    axes[0].set_xlabel("$x$, m")
    axes[0].set_ylabel("$y$, m")
    axes[0].grid(True)

    axes[1].plot(x2_B, y2_B, lw=1.1, color="crimson")
    axes[1].set_title(r"Case B: $\theta_2(0) = \pi/2 + 0.01 + 10^{-6}$")
    axes[1].set_xlabel("$x$, m")
    axes[1].grid(True)

    plt.suptitle("Trajectory divergence under small initial perturbation")
    plt.tight_layout()
    plt.show()

    # 2.3. Relative energy error at 1e-8 vs 1e-5
    sol_A_coarse = solve_ivp(
        rhs, t_span, y0_A, method="RK45", t_eval=t_eval,
        args=(L1, L2, m1, m2, g), rtol=1e-5, atol=1e-5
    )

    E_fine = total_energy(sol_A.y, L1, L2, m1, m2, g)
    rel_err_fine = np.abs(E_fine - E_fine[0]) / np.abs(E_fine[0])

    E_coarse = total_energy(sol_A_coarse.y, L1, L2, m1, m2, g)
    rel_err_coarse = np.abs(E_coarse - E_coarse[0]) / np.abs(E_coarse[0])

    plt.figure(figsize=(9, 4.5))
    plt.plot(t_eval, rel_err_fine, label="rtol = atol = $10^{-8}$", color="green")
    plt.plot(t_eval, rel_err_coarse, label="rtol = atol = $10^{-5}$", color="orange")
    plt.yscale("log")
    plt.title("Relative change in total energy $|E(t) - E(0)| / |E(0)|$")
    plt.xlabel("Time $t$, s")
    plt.ylabel("Relative error (log scale)")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

    # --- TASK 1.5: Double pendulum animation ---
    fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
    ax_anim.set_xlim(-(L1 + L2 + 0.2), L1 + L2 + 0.2)
    ax_anim.set_ylim(-(L1 + L2 + 0.2), L1 + L2 + 0.2)
    ax_anim.set_aspect("equal")
    ax_anim.grid(True)
    ax_anim.set_title("Double pendulum animation")

    line, = ax_anim.plot([], [], "o-", lw=2, color="black", markersize=8)
    trace, = ax_anim.plot([], [], "-", lw=1, color="royalblue", alpha=0.6)

    def init():
        line.set_data([], [])
        trace.set_data([], [])
        return line, trace

    def update(frame):
        line.set_data([0, x1_A[frame], x2_A[frame]], [0, y1_A[frame], y2_A[frame]])
        start = max(0, frame - 150)
        trace.set_data(x2_A[start : frame + 1], y2_A[start : frame + 1])
        return line, trace

    ani = FuncAnimation(fig_anim, update, frames=N, init_func=init, interval=20, blit=True)
    plt.show()

if __name__ == "__main__":
    main()