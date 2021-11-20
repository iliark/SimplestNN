import numpy as np


def act(x):
    return 0 if x < 0 else 1


def buy(comfort, price, brightColored):
    x = np.array([comfort, price, brightColored]) #вектор входных значений
    w11 = [0.8, 0.3, -1]
    w12 = [0.3, 0, 2]     #веса на входах нейронов скрытого слоя
    weight1 = np.array([w11, w12])
    weight2 = np.array([1, -1]) #веса выходов нейронов скрытого слоя

    sum_hidden = np.dot(weight1, x)  #выполним скалярное произведение входных значений и значений весов на входах скрытого слоя
    print("Значения сумм на нейронах скрытого слоя: " + str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Значения на выходах нейронов скрытого слоя: " + str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)       #Воспользуемся функцией активации
    print("Выходное значение НС: " + str(y))

    return y


print("Вводите 1 для положительного ответа, 0 для отрицательного")

comfort = int(input("Салон комфортабельный?: "))
price = int(input("Цена приемлемая?: "))
brightColored = int(input("Яркий цвет?: "))


if buy(comfort, price, brightColored) == 1:
    print("Машина подойдет для меня")
else:
    print("Машина подойдет жене")