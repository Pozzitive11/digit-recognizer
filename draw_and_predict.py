import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image, ImageDraw

class DigitDrawer:
    """
    Інтерактивний інструмент для малювання цифр та їх розпізнавання
    """
    
    def __init__(self, model):
        self.model = model
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Створення полотна для малювання (280x280 для кращої точності)
        self.canvas_size = 280
        self.canvas = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.canvas)
        
        # Параметри малювання
        self.brush_size = 20
        self.drawing = False
        self.last_pos = None
        
        # Налаштування UI
        self.setup_ui()
        
    def setup_ui(self):
        """Налаштування інтерфейсу користувача"""
        # Полотно для малювання
        self.axes[0].set_title('Намалюйте цифру тут', fontsize=14, fontweight='bold')
        self.canvas_display = self.axes[0].imshow(np.zeros((self.canvas_size, self.canvas_size)), 
                                                   cmap='gray', vmin=0, vmax=255)
        self.axes[0].axis('off')
        
        # Графік передбачень
        self.axes[1].set_title('Передбачення моделі', fontsize=14, fontweight='bold')
        self.bars = self.axes[1].bar(range(10), [0]*10, color='lightblue')
        self.axes[1].set_xlabel('Цифра', fontsize=12)
        self.axes[1].set_ylabel('Ймовірність (%)', fontsize=12)
        self.axes[1].set_xticks(range(10))
        self.axes[1].set_ylim(0, 100)
        self.axes[1].grid(axis='y', alpha=0.3)
        
        # Текст з передбаченням
        self.prediction_text = self.axes[1].text(0.5, 1.05, '', 
                                                  transform=self.axes[1].transAxes,
                                                  fontsize=16, fontweight='bold',
                                                  ha='center', va='bottom')
        
        # Кнопки
        clear_ax = plt.axes([0.35, 0.02, 0.1, 0.05])
        predict_ax = plt.axes([0.55, 0.02, 0.1, 0.05])
        
        self.clear_button = Button(clear_ax, 'Очистити')
        self.clear_button.on_clicked(self.clear_canvas)
        
        self.predict_button = Button(predict_ax, 'Розпізнати')
        self.predict_button.on_clicked(self.predict)
        
        # Обробники подій миші
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        
    def on_press(self, event):
        """Обробка натискання кнопки миші"""
        if event.inaxes == self.axes[0]:
            self.drawing = True
            if event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                self.last_pos = (x, y)
                self.draw_point(x, y)
    
    def on_release(self, event):
        """Обробка відпускання кнопки миші"""
        self.drawing = False
        self.last_pos = None
    
    def on_motion(self, event):
        """Обробка руху миші"""
        if self.drawing and event.inaxes == self.axes[0]:
            if event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                
                # Малювання лінії від останньої позиції
                if self.last_pos is not None:
                    self.draw.line([self.last_pos, (x, y)], 
                                   fill=255, width=self.brush_size)
                
                self.draw_point(x, y)
                self.last_pos = (x, y)
    
    def draw_point(self, x, y):
        """Малювання точки на полотні"""
        # Малювання кола
        r = self.brush_size // 2
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
        
        # Оновлення відображення
        self.update_canvas_display()
    
    def update_canvas_display(self):
        """Оновлення відображення полотна"""
        canvas_array = np.array(self.canvas)
        self.canvas_display.set_data(canvas_array)
        self.fig.canvas.draw_idle()
    
    def clear_canvas(self, event=None):
        """Очищення полотна"""
        self.canvas = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.canvas)
        self.update_canvas_display()
        
        # Очищення передбачень
        for bar in self.bars:
            bar.set_height(0)
        self.prediction_text.set_text('')
        
        self.fig.canvas.draw_idle()
    
    def preprocess_canvas(self):
        """Підготовка зображення для моделі (з центруванням як в MNIST)"""
        img_array = np.array(self.canvas)
        
        # Знайти bounding box цифри (де є білі пікселі)
        rows = np.any(img_array > 30, axis=1)
        cols = np.any(img_array > 30, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # Якщо нічого не намальовано
            img_resized = np.zeros((28, 28))
        else:
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Вирізати цифру з невеликим padding
            padding = 20
            rmin = max(0, rmin - padding)
            rmax = min(img_array.shape[0], rmax + padding)
            cmin = max(0, cmin - padding)
            cmax = min(img_array.shape[1], cmax + padding)
            
            cropped = img_array[rmin:rmax, cmin:cmax]
            
            # Визначити розмір для масштабування (зберігаємо пропорції)
            h, w = cropped.shape
            if h > w:
                new_h = 20
                new_w = int(20 * w / h)
            else:
                new_w = 20
                new_h = int(20 * h / w)
            
            # Масштабувати
            img_pil = Image.fromarray(cropped)
            img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img_resized = np.array(img_resized)
            
            # Створити 28x28 зображення та відцентрувати
            final_img = np.zeros((28, 28))
            y_offset = (28 - new_h) // 2
            x_offset = (28 - new_w) // 2
            final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
            img_resized = final_img
        
        # Нормалізація
        img_normalized = img_resized / 255.0
        
        # Зміна форми для моделі
        img_input = img_normalized.reshape(1, 28, 28, 1)
        
        return img_input, img_normalized
    
    def predict(self, event=None):
        """Виконання передбачення"""
        # Перевірка чи є щось намальоване
        canvas_array = np.array(self.canvas)
        if np.sum(canvas_array) == 0:
            self.prediction_text.set_text('Спочатку намалюйте цифру!')
            self.prediction_text.set_color('red')
            self.fig.canvas.draw_idle()
            return
        
        # Підготовка зображення
        img_input, _ = self.preprocess_canvas()
        
        # Передбачення
        predictions = self.model.predict(img_input, verbose=0)[0]
        predicted_digit = np.argmax(predictions)
        confidence = predictions[predicted_digit] * 100
        
        # Оновлення графіку
        for i, bar in enumerate(self.bars):
            bar.set_height(predictions[i] * 100)
            bar.set_color('green' if i == predicted_digit else 'lightblue')
        
        # Оновлення тексту
        self.prediction_text.set_text(f'Це цифра: {predicted_digit} ({confidence:.1f}%)')
        self.prediction_text.set_color('green' if confidence > 80 else 'orange')
        
        self.fig.canvas.draw_idle()
        
        # Виведення в консоль
        print("\n" + "="*50)
        print(f"Передбачена цифра: {predicted_digit}")
        print(f"Впевненість: {confidence:.2f}%")
        print("\nЙмовірності:")
        for i in range(10):
            bar_chart = "█" * int(predictions[i] * 30)
            print(f"  {i}: {predictions[i]*100:6.2f}% {bar_chart}")
        print("="*50)
    
    def show(self):
        """Відображення вікна"""
        plt.show()


def main():
    """Головна функція програми"""
    print("="*60)
    print("Інтерактивне малювання та розпізнавання цифр")
    print("="*60)
    
    # Завантаження моделі
    print("\nЗавантаження моделі...")
    try:
        model = tf.keras.models.load_model('mnist_cnn_model.keras')
        print("✓ Модель успішно завантажена")
    except Exception as e:
        print(f"✗ Помилка при завантаженні моделі: {e}")
        print("\nПереконайтеся, що файл 'mnist_cnn_model.keras' існує.")
        print("Якщо моделі немає, спочатку запустіть: python mnist_cnn.py")
        return
    
    print("\n" + "="*60)
    print("Інструкція:")
    print("  1. Намалюйте цифру (0-9) мишкою на лівому полі")
    print("  2. Натисніть кнопку 'Розпізнати' для передбачення")
    print("  3. Натисніть кнопку 'Очистити' для нового малювання")
    print("="*60)
    print("\nВідкриваю вікно для малювання...")
    
    # Створення та відображення інтерфейсу
    drawer = DigitDrawer(model)
    drawer.show()


if __name__ == "__main__":
    main()

