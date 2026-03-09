import numpy as np
import matplotlib.pyplot as plt


x_data = np.arange(1, 25)
y_data = np.array([-2.0, 3.5, 4.1, 5.15, 6.2, 7.23, 8.22, 9.17, 10.1, 11.5,
                   13.1, 14.3, 15.7, 16.13, 17.19, 18.2, 19.22, 20.21,
                   21.18, 22.15, 23.1, 24.3, 25.0, 26.1])


def solve_gauss(A, B):
    n = len(B)
    M = np.hstack((A, B.reshape(-1, 1)))
    for i in range(n):
        max_el = abs(M[i:, i]).argmax() + i
        M[[i, max_el]] = M[[max_el, i]]
        for j in range(i + 1, n):
            ratio = M[j, i] / M[i, i]
            M[j, i:] -= ratio * M[i, i:]

    res = np.zeros(n)
    for i in range(n - 1, -1, -1):
        res[i] = (M[i, n] - np.dot(M[i, i + 1:n], res[i + 1:n])) / M[i, i]
    return res


def get_mnk_coeffs(x, y, m):
    n = len(x)
    A = np.zeros((m + 1, m + 1))
    B = np.zeros(m + 1)
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x ** (i + j))
        B[i] = np.sum(y * (x ** i))
    return solve_gauss(A, B)


def poly_val(coeffs, x):
    return sum(c * (x ** i) for i, c in enumerate(coeffs))



variances = []
m_range = range(1, 11)
for m in m_range:
    c = get_mnk_coeffs(x_data, y_data, m)
    y_pred = poly_val(c, x_data)
    delta = np.sqrt(np.sum((y_pred - y_data) ** 2) / (len(x_data) + 1))
    variances.append(delta)

opt_m = m_range[np.argmin(variances)]
final_coeffs = get_divided_diff = get_mnk_coeffs(x_data, y_data, opt_m)

# ГРАФІК 1: Апроксимація та прогноз на 3 місяці
plt.figure(figsize=(10, 5))
x_fine = np.linspace(1, 27, 200)
plt.plot(x_data, y_data, 'ro', label='Фактичні дані')
plt.plot(x_fine, poly_val(final_coeffs, x_fine), 'b-', label=f'МНК (ступінь m={opt_m})')
plt.axvspan(24, 27, color='yellow', alpha=0.2, label='Прогноз (3 міс.)')
plt.title('Графік 1: Апроксимація температур та прогноз')
plt.xlabel('Місяць');
plt.ylabel('Температура');
plt.legend();
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
error = y_data - poly_val(final_coeffs, x_data)
plt.stem(x_data, error, linefmt='g-', markerfmt='go', basefmt='k-')
plt.title('Графік 2: Похибка апроксимації (ε = y - φ(x))')
plt.xlabel('Місяць');
plt.ylabel('Похибка');
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(m_range, variances, 'ms-')
plt.plot(opt_m, min(variances), 'ko', markersize=10, label=f'Оптимум (m={opt_m})')
plt.title('Графік 3: Залежність дисперсії від ступеня многочлена')
plt.xlabel('Ступінь m');
plt.ylabel('Дисперсія δ');
plt.legend();
plt.grid(True)
plt.show()

print(f"Оптимальний ступінь многочлена: {opt_m}")
print(f"Коефіцієнти: {final_coeffs}")
