from pathlib import Path
import csv
import math
import random
import sys
import matplotlib.pyplot as plt

TABLE = "var4.csv"
BASE_DIR = Path(__file__).parent
IMG_DIR = BASE_DIR / "images"
OUT_FILE = BASE_DIR / "output.txt"
IMG_DIR.mkdir(exist_ok=True)


def read_distribution(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    y_values = [int(v) for v in rows[0][1:]]
    x_values = []
    matrix = []
    missing = None

    for i, row in enumerate(rows[1:]):
        x_values.append(int(row[0]))
        p_row = []
        for j, cell in enumerate(row[1:]):
            if cell.strip() == "":
                p_row.append(None)
                missing = (i, j)
            else:
                p_row.append(float(cell))
        matrix.append(p_row)
    return x_values, y_values, matrix, missing


def f(num, digits=4):
    if num is None:
        return "?"
    s = f"{num:.{digits}f}"
    return str(float(s))


def print_table(headers, rows, digits=4):
    text_rows = []
    for row in rows:
        text_rows.append([f(v, digits) if isinstance(v, float) else str(v) for v in row])

    all_rows = [list(map(str, headers))] + text_rows
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]

    for k, row in enumerate(all_rows):
        print(" | ".join(row[i].rjust(widths[i]) for i in range(len(headers))))
        if k == 0:
            print("-+-".join("-" * w for w in widths))


def prefix_sum(probs):
    s = 0
    return [(s := s + matrix) for matrix in probs]


def expected_value(values, probs):
    return sum(values[i] * probs[i] for i in range(len(values)))


def variance(values, probs):
    m = expected_value(values, probs)
    return sum((values[i] - m) ** 2 * probs[i] for i in range(len(values)))


def median(values, probs):
    s = 0
    for value, matrix in zip(values, probs):
        s += matrix
        if s >= 0.5:
            return value


def modes(values, probs):
    max_p = max(probs)
    return [values[i] for i in range(len(values)) if abs(probs[i] - max_p) < 1e-12]


def bar_plot(values, probs, title, xlabel, filename):
    plt.figure(figsize=(9, 4.8))
    plt.bar([str(v) for v in values], probs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Вероятность")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=180)
    plt.close()


def cdf_plot(values, cdf_values, title, filename):
    xs = [values[0] - 1] + values
    ys = [0] + cdf_values
    plt.figure(figsize=(9, 4.8))
    plt.step(xs, ys, where="post")
    plt.scatter(values, cdf_values)
    plt.ylim(-0.03, 1.03)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("F(t)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=180)
    plt.close()


def generate_random(values, cdf_values, n):
    result = []
    for _ in range(n):
        u = random.random()
        for value, border in zip(values, cdf_values):
            if u <= border:
                result.append(value)
                break
    return result


def relative_frequencies(values, data):
    n = len(data)
    return [sum(1 for x in data if x == v) / n for v in values]


sys.stdout = open(OUT_FILE, "w", encoding="utf-8")
# Блок 1

x_values, y_values, matrix, missing = read_distribution(BASE_DIR / TABLE)
headers = ["X\\Y"] + y_values

print("Задание 1. Исходная таблица")
print_table(headers, [[x_values[i]] + matrix[i] for i in range(len(x_values))])
print()

# Задание 2. Пропущенная вероятность.
known_sum = sum(cell for row in matrix for cell in row if cell is not None)
mi, mj = missing
matrix[mi][mj] = 1 - known_sum

print("Задание 2. Пропущенная вероятность")
print(f"Сумма известных вероятностей = {f(known_sum)}")
print(f"P(X={x_values[mi]}, Y={y_values[mj]}) = {f(matrix[mi][mj])}")
print(f"Сумма p_ij = {f(sum(sum(row) for row in matrix))}")
print()

# Задание 3. Совместная функция распределения F(x, y).
f_xy = []
for i in range(len(x_values)):
    row = []
    for j in range(len(y_values)):
        s = 0
        for r in range(i + 1):
            for c in range(j + 1):
                s += matrix[r][c]
        row.append(s)
    f_xy.append(row)

print("Задание 3. Таблица F_XY(x_i, y_j)")
print_table(headers, [[x_values[i]] + f_xy[i] for i in range(len(x_values))])
print()

# Задание 4. Маргинальные распределения.
p_x = [sum(row) for row in matrix]
p_y = [sum(matrix[i][j] for i in range(len(x_values))) for j in range(len(y_values))]
f_x = prefix_sum(p_x)
f_y = prefix_sum(p_y)

print("Задание 4")
print("Маргинальное распределение X")
print_table(["x", "P_X(x)", "F_X(x)"], [[x_values[i], p_x[i], f_x[i]] for i in range(len(x_values))])
print()

print("Маргинальное распределение Y")
print_table(["y", "P_Y(y)", "F_Y(y)"], [[y_values[j], p_y[j], f_y[j]] for j in range(len(y_values))])
print()

bar_plot(x_values, p_x, "Закон распределения X", "x", "pmf_x.png")
bar_plot(y_values, p_y, "Закон распределения Y", "y", "pmf_y.png")
cdf_plot(x_values, f_x, "Функция распределения X", "cdf_x.png")
cdf_plot(y_values, f_y, "Функция распределения Y", "cdf_y.png")

# Задание 5. Числовые характеристики.
ex = expected_value(x_values, p_x)
ey = expected_value(y_values, p_y)

dx = variance(x_values, p_x)
dy = variance(y_values, p_y)

x_median = median(x_values, p_x)
y_median = median(y_values, p_y)

x_mode = modes(x_values, p_x)
y_mode = modes(y_values, p_y)

print("Задание 5. Числовые характеристики")
print_table(
    ["Величина", "E", "D", "sigma", "медиана", "мода"],
    [
        [
            "X",
            ex,
            dx,
            math.sqrt(dx),
            x_median,
            ", ".join(map(str, x_mode)),
        ],
        [
            "Y",
            ey,
            dy,
            math.sqrt(dy),
            y_median,
            ", ".join(map(str, y_mode)),
        ],
    ],
)
print()

# Задание 6. Независимость.
max_error = 0
for i in range(len(x_values)):
    for j in range(len(y_values)):
        max_error = max(max_error, abs(matrix[i][j] - p_x[i] * p_y[j]))

print("Задание 6. Независимость")
print(f"max |p_ij - P_X(x_i)P_Y(y_j)| = {f(max_error, 5)}")
print("Вывод: X и Y независимы." if max_error < 1e-12 else "Вывод: X и Y зависимы.")
print()

# Задание 7. Корреляция.
ex_y = 0
for i in range(len(x_values)):
    for j in range(len(y_values)):
        ex_y += x_values[i] * y_values[j] * matrix[i][j]

cov = ex_y - ex * ey
r = cov / (math.sqrt(dx) * math.sqrt(dy))

print("Задание 7. Корреляция")
print(f"E(XY) = {f(ex_y)}")
print(f"cov(X,Y) = {f(cov)}")
print(f"r = {f(r)}")
print()

# Задание 8. Регрессия Y на X: E(Y | X = x).
regression = []
for i in range(len(x_values)):
    conditional_mean = sum(y_values[j] * matrix[i][j] / p_x[i] for j in range(len(y_values)))
    regression.append(conditional_mean)

print("Задание 8. Регрессия Y на X")
print_table(["x", "E(Y|X=x)"], [[x_values[i], regression[i]] for i in range(len(x_values))])
print()


# Блок 2

n_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
N = max(n_values)

# Моделируем X по его маргинальному распределению
data = generate_random(x_values, f_x, N)
freq = relative_frequencies(x_values, data)

print(f"Блок 2")
print(f"Относительные частоты для X, N={N}")

print_table(
    ["x", "P_X(x)", "частота", "разность"],
    [[x_values[i], p_x[i], freq[i], abs(p_x[i] - freq[i])] for i in range(len(x_values))],
)
print()

# Сравнение теоретического и смоделированного распределения
pos = list(range(len(x_values)))
width = 0.38

plt.figure(figsize=(9, 4.8))
plt.bar([v - width / 2 for v in pos], p_x, width=width, label="Теория")
plt.bar([v + width / 2 for v in pos], freq, width=width, label="Моделирование")
plt.xticks(pos, [str(v) for v in x_values])
plt.title("Теория и моделирование для X")
plt.xlabel("x")
plt.ylabel("Вероятность / относительная частота")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(IMG_DIR / "simulation_compare_x.png", dpi=180)
plt.close()

# Сходимость выборочного среднего к E(X)
sample_means = []

for n in n_values:
    sample_means.append(sum(data[:n]) / n)

print("Выборочные средние")
print_table(
    ["n", "среднее"],
    [[n_values[i], sample_means[i]] for i in range(len(n_values))]
)
print()

plt.figure(figsize=(9, 4.8))
plt.plot(n_values, sample_means, marker="o", label="Выборочное среднее")
plt.axhline(ex, linestyle="--", label=f"M(X) = {f(ex)}")
plt.xscale("log")
plt.title("Сходимость выборочного среднего")
plt.xlabel("n")
plt.ylabel("среднее")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(IMG_DIR / "mean_convergence_x.png", dpi=180)
plt.close()

sys.stdout.close()