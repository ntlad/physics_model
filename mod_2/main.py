import math
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as backend_tkagg

def compute_field(n, r, zs):
    # см. Appendix A. Numerical Method in "Mathematics of the Faraday Cage"

    # Центры n дисков
    a_list = range(1, n+1)
    unit_roots = np.array([math.e**(2j * math.pi * m/n) for m in a_list])

    # Вектор радиусов
    rr = r*np.ones(unit_roots.shape)

    # Количество членов в разложении
    N = int(max(0, round(4.0 + 0.5 * np.log10(r))))

    # Количество точек выборки на диске
    npts = 3 * N + 2

    # Выборка npts точек на границе окружностей радиуса r с центрами в корнях из единицы.
    a_list = range(1, int(npts+1))
    circ = np.array([math.e**(m * 2j* math.pi/npts) for m in a_list])

    # Это список, содержащий n массивов
    # Каждый массив имеет размерность npts*1,
    # т.е. каждый из них является вектором-столбцом с одной строкой на точку выборки на окружностях
    z_list = [(unit_roots[i] + rr[i] * circ) for i in range(n)]

    # Объединение n массивов друг над другом, образуя вектор-столбец с n*npts строками.
    z = np.concatenate(z_list)

    # Постоянный член
    A = np.concatenate([np.zeros(1), -np.ones(z.shape[0])])

    # Правая часть уравнения
    rhs = np.concatenate([np.zeros(1), -np.log(np.abs(z - zs))])

    for i in range(n):
        B = np.concatenate([np.ones(1), np.log(np.abs(z-unit_roots[i]))])
        A = np.column_stack((A, B)) # Логарифмические члены
        for k in range(N):
            zck = np.power((z - unit_roots[i]), -(k+1))
            C = np.concatenate([np.zeros(1), zck.real])
            D = np.concatenate([np.zeros(1), zck.imag])
            A = np.column_stack((A, C, D)) # Алгебраические члены

    # Это переопределенная система, подгонка решения методом наименьших квадратов
    x, residuals, rank, s = np.linalg.lstsq(A, rhs, rcond=None)
    x = np.delete(x, (0), axis=0)  # Удаление первой строки

    # Коэффициенты логарифмических членов
    d = x[0:: 2 * N + 1]
    x = np.delete(x, np.s_[0::2*N+1], None)

    # Коэффициенты алгебраических членов
    a = x[0::2]
    rhs = x[1::2]

    # Задание графика
    X = np.linspace(-1.4*zs, 2.2*zs, 500)
    Y = np.linspace(-1.8*zs, 1.8*zs, 500)
    [xx, yy] = np.meshgrid(X, Y)

    zz = xx + 1j*yy
    uu = np.log(np.abs(zz - zs))

    for j in range(n):
        uu = uu + d[j]*np.log(np.abs(zz - unit_roots[j]))
        for k in range(N):
            zck = np.power((zz - unit_roots[j]), -(k+1))
            kk = k + j * N
            uu = uu + a[kk] * zck.real + rhs[kk] * zck.imag
    for j in range(n):
        uu[np.abs(zz - unit_roots[j]) <= rr[j]] = np.nan

    return xx, yy, uu


class Boxes(object):
    def __init__(self, root, n, r, zs):
        self.root = root
        self.n = n
        self.n_entry = tk.Entry(self.root, textvariable=self.n, width=3)
        self.n_label = tk.Label(self.root, text="Количество дисков")
        self.n_label.grid(row=0)

        self.r = r
        self.r_entry = tk.Entry(self.root, textvariable=self.r, width=3)
        self.r_label = tk.Label(self.root, text="Радиус диска")
        self.r_label.grid(row=1)

        self.zs = zs
        self.zs_entry = tk.Entry(self.root, textvariable=self.zs, width=3)
        self.zs_label = tk.Label(self.root, text="Координата источника")
        self.zs_label.grid(row=2)

        self.n_entry.grid(row=0, column=1)
        self.r_entry.grid(row=1, column=1)
        self.zs_entry.grid(row=2, column=1)

class FaradayCage(tk.Frame):
    def __init__(self, parent):
        self.parent = parent
        self.parent.wm_title("Клетка Фарадея")

        # Создание фрейма
        self.frame = tk.Frame(self.parent)
        self.frame.grid(row=4, column=0, columnspan=5)

        # Создание рисунка
        self.fig = plt.figure(figsize=(5, 4), dpi=125)

        # Создание подграфика, в котором будет модель клетки
        self.sub_plt = self.fig.add_subplot(111)
        self.sub_plt.set_aspect('equal')

        # Создание холста и размещение его во фрейме
        self.canvas = backend_tkagg.FigureCanvasTkAgg(self.fig, self.frame)
        self.canvas.get_tk_widget().pack()

        # Определение переменных моделирования
        # n: количество проволок/дисков
        # r: диаметр проволоки/диска
        # zs: координата заряда на оси x
        self.n = tk.IntVar()
        self.r = tk.DoubleVar()
        self.zs = tk.DoubleVar()

        # Начальные значения переменных
        self.n.set(10)
        self.r.set(0.01)
        self.zs.set(2.0)

        # Создание полей ввода и привязка их к переменным моделирования
        self.boxes = Boxes(
            self.parent,
            self.n,
            self.r,
            self.zs
        )

        # Добавление кнопки для построения графика потенциала
        self.plot_ptn_btn = tk.Button(
            self.parent,
            text="Смоделировать",
            command=lambda: self.potential_modeling(),
            width=12
        )
        self.plot_ptn_btn.grid(row=1, column=2)

        self.toolbar = backend_tkagg.NavigationToolbar2Tk(self.canvas, self.frame)

    def potential_modeling(self):
        # Очистка подграфика
        self.clear_plot()

        self.n_value = self.n.get()
        self.r_value = self.r.get()
        self.zs_value = self.zs.get()

        # Построение дисков
        wire_lst = range(1, self.n_value+1)
        unit_roots = np.array([math.e**(2j*math.pi*m/self.n_value) for m in wire_lst])
        self.sub_plt.scatter(unit_roots.real, unit_roots.imag, color='pink')

        # Построение точечного заряда
        self.sub_plt.plot(self.zs_value.real, self.zs_value.imag, '.r')

        self.xx, self.yy, self.uu = compute_field(self.n_value, self.r_value, self.zs_value)

        # Построение линий потенциала
        self.sub_plt.contour(
            self.xx,
            self.yy,
            self.uu,
            levels=np.arange(-2, 2, 0.1),
            colors=('black'),
            corner_mask=True
        )

        # Перерисовка холста
        self.canvas.draw()

    def clear_plot(self):
        self.sub_plt.cla()
        self.canvas.draw()

    def quit(self):
        self.parent.quit()
        self.parent.destroy()

FaradayCage(tk.Tk())
tk.mainloop()
