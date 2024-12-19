import numpy
import matplotlib.pyplot as pyplot

# Задание констант
M = 2150
m = 150
mu = 15
g = 1.62
v_0 = 61
H_0 = 950
v_r = 3660
v_land = 3

def v(t): #создадим функцию возвращающую скорость реактивного торможения
    return v_r * numpy.log(M / (M - mu*t)) - g*t

t_land = 0
v_softland = 3
v_preland = 82.46
current_fuel = m

while True:
    t_land += 0.0005
    t = numpy.linspace(0, t_land)
    H_land = -numpy.trapz(v(t) - v_preland, t)

    if H_land < 0:
        print("Ошибка: высота становится отрицательной, посадка невозможна.")
        break

    if H_land > H_0:
        print("Ошибка: высота начала торможения больше начальной высоты.")
        break

    # Рассчет времени до торможения:
    t_preland = (numpy.sqrt(v_0**2 + 2*g*(H_0 - H_land)) - v_0) / g

    if t_preland < 0:
        print("Ошибка: вычисленное время до торможения отрицательное, посадка невозможна.")
        break

    v_preland = v_0 + g*t_preland
    v_land = v(t_land) - v_preland

    # Cкорость около поверхности < Cкорости мягкой посадки (3м/с)
    if abs(v_land) < v_softland: break

# Проверка, достаточно ли топлива для завершения посадки
time_required_for_landing = t_land + t_preland
total_fuel_required = mu * time_required_for_landing
print(total_fuel_required, mu, time_required_for_landing)

if total_fuel_required > m:
    print("Топлива не хватит для завершения посадки.")
else:
    print('Топлива хватит для завершения посадки.')

print('Высота начала торможения - ', H_land, 'м')
print('Cкорость вблизи поверхности - ', -v_land, 'м/с')

# Отрисовка графика зависимости скорости от времени:
t1 = numpy.linspace(0, t_preland, 30)
t2 = numpy.linspace(t_preland, t_preland + t_land, 30)
v1 = -v_0 - g*t1
v2 = v(t2 - t_preland)-v_preland
fig = pyplot.figure()
pyplot.plot(numpy.hstack([t1, t2]), numpy.hstack([v1, v2]))
ax = pyplot.gca()
ax.grid()
ax.set_xlabel('t, с', size = 'large', weight = 'semibold')
ax.set_ylabel('v м/с', size = 'large', weight = 'semibold')
ax.set_title('Зависимость скорости от времени', size = 'large', weight = 'semibold')
pyplot.show()

# Отрисовка графика зависимости координаты от времени:
H1 = H_0 - v_0*t1 - g*t1**2/2
H2 = []
t2 = numpy.linspace(t_preland, t_preland + t_land, 30)
for i in range(30):
    t = numpy.linspace(0, t2[i] - t_preland)
    dH = numpy.trapz(v(t) - v_preland, t)
    H2.append(H_land + dH)

fig = pyplot.figure()
pyplot.plot(numpy.hstack([t1, t2]), numpy.hstack([H1, H2]))
ax = pyplot.gca()
ax.grid()
ax.set_xlabel('t, с', size = 'large', weight = 'semibold')
ax.set_ylabel('y, м', size = 'large', weight = 'semibold')
ax.set_title('Зависимость координаты от времени', size = 'large', weight = 'semibold')
pyplot.show()

# Отрисовка графика зависимости координаты от времени:
a1 = [-g for i in range(30)]
a2 = ((M - mu*(t2 - t_preland))*g + v_r*mu)/(M - mu*(t2 - t_preland))

fig = pyplot.figure()
pyplot.plot(numpy.hstack([t1, t2]), numpy.hstack([a1, a2]))
ax = pyplot.gca()
ax.grid()
ax.set_xlabel('t, с', size = 'large', weight = 'semibold')
ax.set_ylabel('a, м', size = 'large', weight = 'semibold')
ax.set_title('Зависимость вертикального ускорения от времени', size = 'large', weight = 'semibold')
pyplot.show()
