import numpy as np
import matplotlib.pyplot as plt

# --- 1. Вспомогательные функции активации ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # Предполагается, что x = sigmoid(x)

# --- 2. Подготовка обучающих данных (XOR) ---
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# --- 3. Инициализация гиперпараметров ---
np.random.seed(42)
input_size = 2
hidden_size = 4
output_size = 1
lr = 0.1
epochs = 10000

# --- 4. Инициализация весов ---
weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
bias_hidden = np.zeros((1, hidden_size))

weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
bias_output = np.zeros((1, output_size))

loss_history = []

# --- 5. Обучение ---
for epoch in range(epochs):
    # FORWARD
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output = sigmoid(final_input)

    # LOSS (Binary Cross Entropy)
    loss = np.mean((y - output) ** 2)
    loss_history.append(loss)

    # BACKPROPAGATION
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Обновление весов и смещений
    weights_hidden_output += hidden_output.T.dot(output_delta) * lr
    bias_output += np.sum(output_delta, axis=0, keepdims=True) * lr

    weights_input_hidden += X.T.dot(hidden_delta) * lr
    bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * lr

    # Печатаем каждые 1000 эпох
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# --- 6. Визуализация потерь ---
plt.plot(loss_history)
plt.title("Функция потерь (Loss) во время обучения")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# --- 7. Проверка предсказаний ---
print("\nПроверка предсказаний:")
for i in range(len(X)):
    pred = sigmoid(np.dot(sigmoid(np.dot(X[i], weights_input_hidden) + bias_hidden), weights_hidden_output) + bias_output)
    print(f"{X[i]} => {pred.round(3)}")
