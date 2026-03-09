import numpy as np
import matplotlib.pyplot as plt

x_v5 = np.array([50, 200, 400, 800, 1600])
y_v5 = np.array([120, 110, 90, 65, 40])


def get_divided_diff(x, y):
    size = len(y)
    coef = np.copy(y).astype(float)
    for j in range(1, size):
        for i in range(size - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])
    return coef


def newton_calc(coef, x_knots, x_val):
    order = len(x_knots) - 1
    res = coef[order]
    for k in range(1, order + 1):
        res = coef[order - k] + (x_val - x_knots[order - k]) * res
    return res


def find_limit_objects(coef, x_knots, target_fps=60):
    a, b = 800, 1600
    for _ in range(50):
        mid = (a + b) / 2
        if newton_calc(coef, x_knots, mid) > target_fps:
            a = mid
        else:
            b = mid
    return a


nodes_to_test = [5, 10, 20]
x_smooth = np.linspace(min(x_v5), max(x_v5), 200)

for count in nodes_to_test:
    plt.figure(figsize=(8, 5))
    if count > len(x_v5):
        x_nodes = np.linspace(x_v5[0], x_v5[-1], count)
        y_nodes = np.interp(x_nodes, x_v5, y_v5)
    else:
        indices = np.linspace(0, len(x_v5) - 1, count, dtype=int)
        x_nodes, y_nodes = x_v5[indices], y_v5[indices]

    current_coefs = get_divided_diff(x_nodes, y_nodes)
    y_curve = [newton_calc(current_coefs, x_nodes, xi) for xi in x_smooth]

    plt.plot(x_v5, y_v5, 'ro', label='Data')
    plt.plot(x_smooth, y_curve, 'b-', label=f'Newton ({count} nodes)')
    plt.axhline(y=60, color='gray', linestyle='--')
    plt.title(f'Nodes: {count}')
    plt.xlabel('Objects');
    plt.ylabel('FPS')
    plt.legend();
    plt.grid(True)
    plt.show()

final_coefs = get_divided_diff(x_v5, y_v5)
fps_1000 = newton_calc(final_coefs, x_v5, 1000)
max_obj_60fps = find_limit_objects(final_coefs, x_v5)

print(f"FPS for 1000 objects: {fps_1000:.2f}")
print(f"Max objects for 60 FPS: {int(max_obj_60fps)}")
