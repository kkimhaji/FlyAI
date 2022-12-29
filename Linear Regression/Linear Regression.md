# 선형 회귀 Linear Regression

- 독립변수와 종속변수를 선형적인 관계로 가정하고, 데이터를 가장 잘 나타내는 선형식을 찾는다.
    - 독립변수 X: 특성
    - 종속변수 Y: 타깃값, 라벨(분류의 경우)
- **종속 변수 Y를 독립변수 X들의 선형 결합으로 표현한 모델 = 선형 회귀 모델**
    - Y = aX + b
        - a(계수)를 찾아내는 것이 선형회귀의 목적? 동작 방식
    - $y = a_1x_1  + a_2x_2 + ... + a_nx_n$
        - 선형적인 결합이기 때문에 선형 회귀임
- 학습에 종속변수가 필요한 지도학습
- 타깃값(종속변수)가 연속된 숫자값이어야 선형 회귀가 가능

## 단순 선형 회귀 Simple Linear Regression

- 독립변수(X, feature)가 1개인 선형 회귀
- 독립변수 1개, 종속변수 1개
- 좌표 평면에 찍었을 때 데이터를 가장 잘 나타내는 직선 하나를 찾아야 함
- 분류 같은 경우에는 데이터의 그룹을 나누는 선인 결정 경계를 찾는 것이 목적
- 선형 회귀는 데이터들을 가장 잘 설명하는 선을 찾아내는 것이 목적
    - 가장 잘 표현한다: Cost 가 최소가 되는 선

### Cost 계산

- 전체 데이터의 정답과 예측값과의 차이를 계산
- Cost가 최소가 되는 선을 찾음
- 선형 회귀를 푼다 = 데이터를 가장 잘 표현하는 직선을 찾는 것 = Cost가 제일 작은 선을 찾는 것

![image](https://user-images.githubusercontent.com/55172514/209932023-219cc959-2ee8-4251-acca-d08c6b5e4535.png)


파란 선 = 예측 값

### 잔차 제곱합(RSS: Residual Sum of Squared)
![image](https://user-images.githubusercontent.com/55172514/209932086-40964ce6-7e25-4eb8-bfe1-05ce5575388b.png)

![image](https://user-images.githubusercontent.com/55172514/209932122-2987bb66-6edc-4584-84cd-2265452b3973.png)


- 음수 값의 처리를 위해 제곱
- RSS가 가장 작은 경우를 찾으면 = Cost가 제일 작은 직선을 찾을 수 있음

### 최소 제곱법 Ordinary Least Squares, OLS

- RSS를 최소로 하는 값 찾기
- 사이킷런에서 사용됨
- 절편 구하는 공식과 기울기 구하는 공식을 통해서 구할 수 있음
    - 증명 나중에 찾아보기
- x에서 x의 평균을 빼고 y에서 y의 평균을 빼서 곱하고 뭐 …
- 딥러닝에서는 안 먹히는 방법
    - 딥러닝에서는 여러 가지 상황이 발생해서
    - 딥러닝에서는 경사하강법 사용

## 선형 회귀

- 손실 함수를 최소로 하는 파라미터를 찾는 방법
    - 최소 제곱법: 사이킷런(머신러닝 수준)에서 사용, 해석학적 방법
    - 경사하강법: 딥러닝에서 사용, 점진적 학습

# 다항 회귀 Polynomial Regression

- feature가 여러 개인 경우 = 다중 회귀 (≠ 다항 회귀) 사용
    - X가 하나: Simple Linear Regression
    - X가 여러 개: 다중 회귀 Multiple Linear Regression
- 데이터가 단순한 직선이 아닐 경우 사용
    - 다중 선형 회귀: $y = a_1x_1  + a_2x_2 + ... + a_nx_n$ 의 형태로 선형 결합 + 직선
    - 다항 회귀: $y = a_1x_1 +  a_1{x_1}^2$ 등의 형태
        - 제곱형태  → 데이터가 직선이 아님 (곡선)
        - 선형 결합은 맞음(계수들 사이의 관계가 직선이라는 뜻)
- 데이터에 각 특성의 거듭제곱을 새로운 특성으로 추가하고, 선형 모델로 학습 시키는 기법

---

# 규제 Regularization

- 규제를 하는 이유: over-fitting을 막기 위해서
- 알고리즘마다 규제를 하는 방법이 정해져 있음
    - tree: max_depth
    - 선형 회귀도 규제를 해야 함
- 고차 다항 회귀 적용 → 과대 적합을 일으키기 쉬움
- 과대 적합을 일으키는 이유
    - 모델 자체가 너무 복잡
    - 학습하기에 데이터가 너무 적음
- 모델의 규제를 통해서 다항식의 차수를 줄이는 것 (고차항의 계수를 0으로)으로 과대적합을 줄일 수 있다.
![image](https://user-images.githubusercontent.com/55172514/209932174-9ef36ac8-0efd-4fdb-99d9-78c456f82144.png)


overfitting
![image](https://user-images.githubusercontent.com/55172514/209932219-0350a1b6-0d35-4374-b959-521719b5e0c7.png)

- 최소가 되는 ${\theta}_0, {\theta}_1$ 를 찾기
- 손실 함수가 위처럼 생긴 상태에서
- 높은 차수를 줄이기 위해 일부러 $1000 {\theta}_3$, $1000 {\theta}_4$ (규제항)를 추가
    - 증하가는 걸 억제시킴
    - ${\theta}_3, {\theta}_4$ 가 0에 가까울 수록 결정됨

## 릿지 규제 L2 Ridge Regularization

- 계수의 제곱을 모두 더하는 것
- 얼만큼 규제를 할지 정하는 ${\lambda}$ 람다 = 하이퍼 파라미터
![image](https://user-images.githubusercontent.com/55172514/209932327-090078aa-7989-4ece-95df-43f891b25c1e.png)

## 라소 규제 L1, Lasso Regularization

- 계수의 절댓값을 모두 더하는 법
![image](https://user-images.githubusercontent.com/55172514/209932376-3ffd01bb-b2fc-4948-b387-06d77bf10bd9.png)


위 두 규제가 제일 많이 쓰임

---

# 로지스틱 회귀 Logistic Regression

- 회귀 알고리즘이지만 **분류**에 사용
- 선형회귀처럼 입력 특성치의 가중치 합을 계산
- 가중치의 합을 바로 출력X, 결과값의 로지스틱(Logistic)을 출력 → logit 변환
- 이진 분류
    - 이진에만 사용하는 것은 아님
    - 다중 분류에도 사용함

### 데이터를 가장 잘 표현하는 직선 찾기 = Regression

![image](https://user-images.githubusercontent.com/55172514/209932402-8292f263-a62a-4593-ac4c-67bf2a10fa52.png)
- 결과 값이 0 or 1
- 찾고자 하는 직선 = $-{\infin}$에서 ${\infin}$까지 감

## 시그모이드 함수 Sigmoid Function

- 로지스틱 함수라고도 함
![image](https://user-images.githubusercontent.com/55172514/209932427-04a06182-d3fd-4737-97a9-44673c0cf39b.png)


- 에러가 가장 작을 수 있는 곡선을 찾는 문제
- ${\beta}_1$에 따라서 곡선의 높이, ${\beta}_0$에 따라서 기울기가 결정됨
- 선형 회귀 = 직선 찾는 거
- 곡선을 찾기 위해서 ax + b의 값을 변경하며 찾는 거
    - 경사하강법으로 찾아야 함
- 시그모이드 곡선 = 가설 ($\hat{y}$)

### 데이터를 가장 잘 나타내는 시그모이드 곡선 찾기

![image](https://user-images.githubusercontent.com/55172514/209932463-4160112e-5665-4197-8161-deeb2cdc062b.png)

- 곡선이기 때문에 중간에 오면 → 확률로 값이 나옴 (1에 0.7정도로 가깝다)

---

# 다중 분류 Multi-class Classification

![image](https://user-images.githubusercontent.com/55172514/209932491-c04e1856-c89a-4cd0-a4bd-188d4e775142.png)


- 이진 분류를 여러 개
- 이진 분류를 먼저 해서 X인지 아닌지 판별
    - 후에 세모인지 아닌지 / 네모인지 아닌지 판별
    - 한 데이터가 X일 확률 : 아닐 확률 =  0.3 : 0.7
        - 세모에 0.7: 0.3
        - 네모에 0.2 : 0.8
            
            ⇒ 세모일 확률이 제일 높으므로 세모라고 판단함
            
- One vs. Rest
- 이진분류를 다중 분류로 사용하는 방법
    - 로지스틱 회귀를 다중 분류로 사용할 때

![image](https://user-images.githubusercontent.com/55172514/209932529-fbb3f7d9-765a-4084-afee-bbf30896f6c5.png)
