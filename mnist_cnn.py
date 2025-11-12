import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. Завантаження та підготовка даних
print("Завантаження даних MNIST...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Форма тренувальних даних: {x_train.shape}")
print(f"Форма тестових даних: {x_test.shape}")

# 2. Створення моделі CNN
print("\nСтворення моделі CNN...")
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 3. Компіляція моделі
print("Компіляція моделі...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Виведення структури моделі
model.summary()

# 4. Тренування моделі
print("\nПочаток тренування...")
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

# 5. Оцінка моделі
print("\nОцінка моделі на тестовому наборі...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nТочність на тестовому наборі: {accuracy:.4f}")
print(f"Втрати на тестовому наборі: {loss:.4f}")

# Збереження моделі та історії навчання
print("\nЗбереження моделі...")
model.save('mnist_cnn_model.keras')
print("Модель збережена як 'mnist_cnn_model.keras'")

# Збереження історії навчання
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("Історія навчання збережена як 'training_history.pkl'")

