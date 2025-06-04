import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Параметры системы (СИ)
R = 10.0       # Радиус кривизны линзы [м]
lambda0 = 550e-9  # Центральная длина волны [м] (зелёный свет)
delta_lambda = 50e-9  # Ширина спектра [м]
I0 = 1.0       # Максимальная интенсивность [отн. ед.]

# Радиальная координата от -10 до 10 мм с высоким разрешением
r_mm = np.linspace(-10, 10, 5000)  # 5000 точек для гладкости
r = r_mm * 1e-3  # Переводим в метры

# Функции расчёта интенсивности
def intensity_mono(r, lambda_):
    d = r**2 / (2 * R)  # Толщина воздушного зазора
    delta = 2 * d + lambda_/2  # Оптическая разность хода
    return I0 * np.cos(np.pi * delta / lambda_)**2

def intensity_quasi(r, lambda0, delta_lambda, steps=100):
    wavelengths = np.linspace(lambda0 - delta_lambda/2,
                             lambda0 + delta_lambda/2,
                             steps)
    return np.mean([intensity_mono(r, wl) for wl in wavelengths], axis=0)

# Создание 2D сетки (10x10 мм)
x_mm = np.linspace(-10, 10, 1000)
y_mm = np.linspace(-10, 10, 1000)
X_mm, Y_mm = np.meshgrid(x_mm, y_mm)
R_mm_grid = np.sqrt(X_mm**2 + Y_mm**2)  # Радиус в мм
R_grid = R_mm_grid * 1e-3  # Переводим в метры

# Расчёт интенсивностей
I_mono = intensity_mono(r, lambda0)
I_quasi = intensity_quasi(r, lambda0, delta_lambda)
I_mono_2D = intensity_mono(R_grid, lambda0)
I_quasi_2D = intensity_quasi(R_grid, lambda0, delta_lambda)

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
fig = plt.figure(figsize=(20, 6))

# График 1: Зависимость интенсивности от радиуса
ax1 = plt.subplot(1, 3, 1)
plt.plot(r_mm, I_mono, label='Монохроматический', color='navy', linewidth=1.5)
plt.plot(r_mm, I_quasi, label='Квазимонохроматический', color='darkorange', linewidth=1.5)
plt.xlabel('Радиус [мм]', fontsize=12)
plt.ylabel('Интенсивность [отн. ед.]', fontsize=12)
plt.title('Радиальное распределение интенсивности', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-10, 10)

# График 2: 2D карта для монохроматического света
ax2 = plt.subplot(1, 3, 2)
im2 = plt.imshow(I_mono_2D, cmap='inferno',
               extent=[-10, 10, -10, 10],
               aspect='auto',
               vmin=0, vmax=1)
plt.title('Монохроматические кольца', fontsize=14)
plt.xlabel('X [мм]', fontsize=12)
plt.ylabel('Y [мм]', fontsize=12)
cbar2 = plt.colorbar(im2, pad=0.01)
cbar2.set_label('Интенсивность', fontsize=10)

# График 3: 2D карта для квазимонохроматического света
ax3 = plt.subplot(1, 3, 3)
im3 = plt.imshow(I_quasi_2D, cmap='inferno',
               extent=[-10, 10, -10, 10],
               aspect='auto',
               vmin=0, vmax=1)
plt.title('Квазимонохроматические кольца', fontsize=14)
plt.xlabel('X [мм]', fontsize=12)
plt.ylabel('Y [мм]', fontsize=12)
cbar3 = plt.colorbar(im3, pad=0.01)
cbar3.set_label('Интенсивность', fontsize=10)

plt.tight_layout()
plt.show()