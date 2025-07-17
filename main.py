import src.preprocess as pp
import src.train as tr
import src.evaluate as ev

if __name__ == '__main__':

    # Этапы обучения

    #print("1) Препроцессинг данных")
    #pp.preprocess()
    #print("\n2) Обучение модели")
    #tr.train()
    #print("\n3) Оценка на тестовом наборе")
    #ev.evaluate()

    print("\n Вводите текст для проверки токсичности.")
    print(" Для выхода введите 'exit' или 'quit'.")
    while True:
        txt = input("\nВведите текст: ").strip()
        if txt.lower() in ('exit', 'quit'):
            print("Выход.")
            break
        prob = ev.predict_text(txt)
        label = "ТОКСИЧНЫЙ" if prob > 0.8 else "НЕ токсичный"
        print(f"🔹 Вероятность токсичности: {prob:.2f} → {label}")
