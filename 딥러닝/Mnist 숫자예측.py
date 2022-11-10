from tensorflow import keras #라이브러리 로드
import numpy as np
import matplotlib.pyplot as plt

dataset = keras.datasets.mnist #mnist 데이터 로드
(train_X, train_Y), (test_X, test_Y) = dataset.load_data()

train_X, test_X = train_X/255, test_X/255  #데이터 전처리 과정 : 픽셀값을 0과 1사이로 조정하기 위해서 한다.

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28))) #input되는 image의 크기 설정

model.add(keras.layers.Dense(784))
model.add(keras.layers.Dense(1024))
model.add(keras.layers.Dense(512))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()

model.fit(train_X, train_Y, epochs=5, validation_split = 0.2)

test_loss, test_accuracy = model.evaluate(test_X, test_Y)

print('test_loss = %.2f' %test_loss)
print('test_accuracy = %.2f' %test_accuracy)

target_image = test_X[0]
plt.imshow(target_image)
plt.show()

pred = model.predict(target_image.reshape(1, 28, 28))
print(pred)
print("예측 결과, 해당 image는 %d 로 예측 되었습니다." %np.argmax(pred))
