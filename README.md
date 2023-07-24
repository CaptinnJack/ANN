# 앞으로 이루어지는 모든 모델들의 데이터는 라이브러리에서 가져오기에 구글 마운트를 실행하지 않습니다.
# tensorflow 라이브러리 설치
!pip install tensorflow

# 딥러닝 관련 라이브러리 불러오기
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# MNIST 데이터셋 불러오기
mnist = keras.datasets.mnist

# MNIST 데이터셋 학습용(x,y), 테스트용(x,y)으로 나누기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 학습용 데이터 형태 살펴보기
x_train.shape

# 학습용 첫 번째 데이터 살펴보기1
x_train[1]

plt.imshow(x_train[1], cmap='gray')

# 학습용 첫 번째 데이터 살펴보기2
y_train[1]

# 데이터 전처리(0 ~ 1 사이 숫자로)
x_train = x_train / 255
x_test = x_test / 255

# 데이터 전처리 결과 확인
x_train[0]

# 모델 만들기(keras 라이브러리에서 모델을 가져와 설계합니다) : 입력층(1*784) - 은닉층1(256) - 은닉층2(128) - 은닉층3(64) - 출력층(10)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(256, activation = 'sigmoid'),
    keras.layers.Dense(128, activation = 'sigmoid'),
    keras.layers.Dense(64, activation = 'sigmoid'),
    keras.layers.Dense(10, activation = 'softmax')])

# 모델 컴파일 : 최적화 함수, 손실 함수 설정 + 평가 지표 설정 + 가중치 초기화)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# 모델 확인
model.summary()

# 모델 학습 : 전체 데이터는 5번 반복(epochs는 학습 횟수로서 조절이 가능합니다)
model.fit(x_train, y_train, epochs = 5)

# 모델 평가
model.evaluate(x_test, y_test)

# 예측 - 0번째 숫자 이미지로 보기
plt.imshow(x_train[7], cmap='gray')

# 예측 - 0번째 숫자 예측하기1
print(model.predict(x_train[7].reshape(1, 28, 28)))

print(np.argmax(model.predict(x_train[7].reshape(1, 28, 28))))
