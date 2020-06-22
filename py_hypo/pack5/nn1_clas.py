import numpy as np


class MyPerceptron(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):

        self.eta = eta  # 학습률  0.0 ~ 1.0 사이의 값으로 학습에 참여

        self.n_iter = n_iter  # 훈련 dataset 반복 횟수

        self.random_state = random_state  # 난수로 발생되는 가중치 난수 시드 넘버 (seed=1 이런 거)

    # 가중치를 초기화한 후 train dataset에 있는 각 샘플을 반복 순회하면서 퍼셉트론 학습규칙에 따라 가중치를 갱신

    def fit(self, x, y):  # x:훈련데이터, y:target값

        rgen = np.random.RandomState(self.random_state)  # 난수 생성기

        # 정규분포를 따르는 난수를 발생시켜 가중치(weight)로 사용

        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])

        self.errors_ = []  # 반복학습 마다 발생된 분류 오류를 기억할 누적 기억용

        for _ in range(self.n_iter):

            errs = 0

            for xi, target in zip(x, y):

                update = self.eta * (target - self.predict(xi))

                # print('update : ', update)

                self.w_[1:] += update * xi

                self.w_[0] += update  # 절편

                # print('w_[0] : ', self.w_[0])

                errs += int(update != 0.0)

            self.errors_.append(errs)

        return self

    def net_input(self, x):

        # 최종 입력 계산

        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):

        # print('x : ', x) # [5.1 1.4][4.9 1.4] ...

        # 단위 계단함수를 사용하여 클래스 레이블 반환

        return np.where(self.net_input(x) >= 0.0, 1, -1)
