import numpy as np


def act(x):
    return 0 if x < 0 else 1


def buy(comfort, price, brightcolored):
    x = np.array([comfort, price, brightcolored])
    w11 = [0.8, 0.3, -1]
    w12 = [0.3, 0, 2]
    weight1 = np.array([w11, w12])
    weight2 = np.array([1, -1])

    sum_hidden = np.dot(weight1, x)  # вычисляем сумму на входах нейронов скрытого слоя
    print("Значения сумм на нейронах скрытого слоя: " + str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Значения на выходах нейронов скрытого слоя: " + str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print("Выходное значение НС: " + str(y))

    return y


print("Вводите 1 для положительного ответа, 0 для отрицательного")

comfort = int(input("Салон комфортабельный?: "))
price = int(input("Цена приемлемая?: "))
brightcolored = int(input("Яркий цвет?: "))


res = buy(comfort, price, brightcolored)
if res == 1:
    print("Машина подойдет для меня")
else:
    print("Машина подойдет жене")
