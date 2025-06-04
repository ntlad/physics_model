import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Параметры системы
m = 50.0  # масса маятника (кг)
L = 0.79  # длина маятника (м)
L1 = 0.35  # расстояние крепления пружины (м)
k = 100.0  # жёсткость пружины (Н/м)
beta = 0.3  # коэффициент затухания (1/с)
g = 9.81  # ускорение свободного падения (м/с²)

# Начальные условия: theta1, dtheta1/dt, theta2, dtheta2/dt
initial_conditions = [0.17, 0.0, -0.17, 0.0]

# Временной интервал
t = np.linspace(0, 20, 1000)

# Система уравнений
def derivatives(y, t, m, L, L1, k, beta, g):
    theta1, z1, theta2, z2 = y
    omega0_sq = g / L
    coupling = (k * L1 ** 2) / (m * L ** 2)

    dtheta1 = z1
    dz1 = -2 * beta * z1 - omega0_sq * theta1 + coupling * (theta2 - theta1)

    dtheta2 = z2
    dz2 = -2 * beta * z2 - omega0_sq * theta2 - coupling * (theta2 - theta1)

    return [dtheta1, dz1, dtheta2, dz2]

# Решение системы
solution = odeint(derivatives, initial_conditions, t, args=(m, L, L1, k, beta, g))
theta1, omega1, theta2, omega2 = solution.T

# Построение графиков
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, theta1, label='Маятник 1 (угол)')
plt.plot(t, theta2, label='Маятник 2 (угол)')
plt.xlabel('Время (с)')
plt.ylabel('Угол (рад)')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, omega1, label='Маятник 1 (скорость)')
plt.plot(t, omega2, label='Маятник 2 (скорость)')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (рад/с)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()