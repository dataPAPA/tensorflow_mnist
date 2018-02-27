# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
    # Загружаем MNIST датасет - числа, написанные от руки
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # Задаем граф вычислений в тензорфлоу
    # Плейсхолдеры - те места, куда будут подставляться значения входных-выходных переменных
    x = tf.placeholder("float", [None, 784])
    y = tf.placeholder("float", [None, 10])

    # Переменные - это веса нашего единственного слоя сети
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Предсказание это линейное преобразование входного вектора.<br>
    # До преобразования размерность 784
    # После преобразования - 10
    linear_prediction = tf.matmul(x, W) + b
    scaled_prediction = tf.nn.softmax(linear_prediction) # Softmax

    # Функция потерь - кросс энтропия. В двух словах не объясню почему так. 
    # Почитайте лучше википедию. Но она работает
    loss_function = tf.losses.softmax_cross_entropy(y, linear_prediction)

    # Оптимизатор у нас простой градиентный спуск
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

    # Инициализируем сессию, с которой будем работать
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    # Цикл обучения. Учимся на минибатчах, каждые 5 шагов печатаем ошибку
    batch_size = 100

    for iteration in range(30000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, 
                 feed_dict={x: batch_x, y: batch_y})
        if iteration % 5000 == 0:
            loss = loss_function.eval(
                {x: mnist.test.images, 
                 y: mnist.test.labels})
            print ("#{}, loss={:.4f}".format(iteration, loss))

    # Задаем граф вычислений, выдающий точность предсказания
    predicted_label = tf.argmax(scaled_prediction, 1)
    actual_label = tf.argmax(y, 1)
    is_equal_labels = tf.equal(actual_label, predicted_label)
    accuracy = tf.reduce_mean(tf.cast(is_equal_labels, "float"))

    # Вычисляем точность
    accracy_value = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    print ("Accuracy:", accracy_value)

    # Предсказываем лейбы для тествого датасета
    predicted_label = tf.argmax(scaled_prediction, 1)
    predicted_test_values = predicted_label.eval(
        {x: mnist.test.images})
    print ("Predictions: {}".format(predicted_test_values))
