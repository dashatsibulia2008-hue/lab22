import numpy as np
import matplotlib.pyplot as plt
import requests

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.164983,24.523574|48.166053,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
response = requests.get(url).json()
data = response["results"]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def solve_spline(x, y):
    n = len(x)
    h = np.diff(x)
    A = np.zeros((n, n))
    B = np.zeros(n)
    A[0, 0], A[-1, -1] = 1, 1
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        B[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    c = np.linalg.solve(A, B)
    a = y[:-1]
    b = (y[1:] - y[:-1]) / h - h * (c[1:] + 2 * c[:-1]) / 3
    d = (c[1:] - c[:-1]) / (3 * h)
    return a, b, c[:-1], d


def eval_spline(xk, xq, a, b, c, d):
    idx = np.clip(np.searchsorted(xk, xq) - 1, 0, len(a) - 1)
    dx = xq - xk[idx]
    return a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3


elevs = np.array([p['elevation'] for p in data])
lats, lons = np.array([p['latitude'] for p in data]), np.array([p['longitude'] for p in data])
dist = [0.0]
for i in range(1, len(data)):
    dist.append(dist[-1] + haversine(lats[i - 1], lons[i - 1], lats[i], lons[i]))
X_f, Y_f = np.array(dist), elevs

xs_fine = np.linspace(X_f[0], X_f[-1], 1000)
nodes_to_test = [10, 15, 20]
colors = ['orange', 'green', 'red']


all_splines = {}

sa_e, sb_e, sc_e, sd_e = solve_spline(X_f, Y_f)
all_splines['etalon'] = eval_spline(X_f, xs_fine, sa_e, sb_e, sc_e, sd_e)

for n in nodes_to_test:
    idx = np.round(np.linspace(0, len(X_f) - 1, n)).astype(int)
    xk, yk = X_f[idx], Y_f[idx]
    sa, sb, sc, sd = solve_spline(xk, yk)
    all_splines[n] = eval_spline(xk, xs_fine, sa, sb, sc, sd)

plt.figure(figsize=(10, 6))  
plt.plot(xs_fine, all_splines['etalon'], 'b-', linewidth=2, label='21 вузол (еталон)')

for n, color in zip(nodes_to_test, colors):
    plt.plot(xs_fine, all_splines[n], color=color, alpha=0.7, label=f'{n} вузлів')

plt.title('Вплив кількості вузлів на апроксимацію')
plt.ylabel('Висота (м)')
plt.legend()
plt.grid(True)


# --- Графік 2: Похибка апроксимації 
plt.figure(figsize=(10, 6))  

print("\nСтатистика похибок (в консолі):")
for n, color in zip(nodes_to_test, colors):
    
    error = np.abs(all_splines['etalon'] - all_splines[n])

    plt.plot(xs_fine, error, color=color, label=f'{n} вузлів')

    print(f"===== {n} вузлів =====")
    print(f"Максимальна похибка: {np.max(error):.10f}")
    print(f"Середня похибка: {np.mean(error):.10f}")

plt.title('Похибка апроксимації')
plt.xlabel('Відстань (м)')
plt.ylabel('Абсолютна похибка (м)')
plt.legend()
plt.grid(True)
plt.show()
