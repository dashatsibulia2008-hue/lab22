import requests
import numpy as np
import matplotlib.pyplot as plt


# 1. Запит до API (Стор. 5)
def get_data():
    # Встав сюди ПОВНИЙ рядок координат із пункту 1 "Хід роботи"
    coords = "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927"
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={coords}"
    data = requests.get(url).json()
    return data["results"]


# 2. Функція Haversine (Стор. 6)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# 3. МЕТОД ПРОГОНКИ (Стор. 4) - як вимагає лаба
def solve_progonka(x, y):
    n = len(x)
    h = np.diff(x)

    # Формування тридіагональної матриці для коефіцієнтів c
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n)

    for i in range(1, n - 1):
        A[i] = h[i - 1]
        B[i] = 2 * (h[i - 1] + h[i])
        C[i] = h[i]
        D[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Прямий хід прогонки
    alpha = np.zeros(n)
    beta = np.zeros(n)
    for i in range(1, n - 1):
        m = A[i] * alpha[i - 1] + B[i]
        alpha[i] = -C[i] / m
        beta[i] = (D[i] - A[i] * beta[i - 1]) / m

    # Зворотний хід
    c = np.zeros(n)
    for i in range(n - 2, 0, -1):
        c[i] = alpha[i] * c[i + 1] + beta[i]

    # Розрахунок a, b, d через c (Стор. 9)
    a = y[:-1]
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c[:-1], d


# --- Виконання ---
results = get_data()
lats = [p['latitude'] for p in results]
lons = [p['longitude'] for p in results]
elevs = [p['elevation'] for p in results]

dist = [0]
for i in range(1, len(results)):
    dist.append(dist[-1] + haversine(lats[i - 1], lons[i - 1], lats[i], lons[i]))

x, y = np.array(dist), np.array(elevs)
a, b, c, d = solve_progonka(x, y)

# 4. Характеристики маршруту (Стор. 8-9)
total_dist = x[-1]
total_ascent = sum(max(0, y[i] - y[i - 1]) for i in range(1, len(y)))
energy = 80 * 9.81 * total_ascent  # m=80kg, g=9.81

print(f"Загальна довжина: {total_dist:.2f} м")
print(f"Набір висоти: {total_ascent:.2f} м")
print(f"Енергія: {energy / 1000:.2f} кДж")

# 5. Графік (Стор. 8)
plt.plot(x, y, 'ro', label='Точки')
for i in range(len(x) - 1):
    xs = np.linspace(x[i], x[i + 1], 10)
    ys = a[i] + b[i] * (xs - x[i]) + c[i] * (xs - x[i]) ** 2 + d[i] * (xs - x[i]) ** 3
    plt.plot(xs, ys, 'b-')
plt.grid(True)
plt.show()