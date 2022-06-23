# 미로탈출 강화학습 프로젝트

---

## 1. 프로젝트 개요

미로를 해결하는 여러가지 알고리즘이 있다. 하지만 이번 프로젝트에서는 이러한 알고리즘 중에서도, 강화학습 알고리즘을 사용하고자 한다.

강화학습 알고리즘의 Policy Based 방법과 Actor-Critic 방법, DQN(Deep Q learning), 그리고 마지막으로 PPO(Proximal Policy Optimization)을 사용하여 리턴(cumulative rewards)을 비교하고자 한다. 그리고 학습된 알고리즘의 성능을 쉽게 확인할 수 있는 프로그램을 pygame을 사용하여 작성하면서 프로젝트를 마무리하고자 한다.

이 프로젝트는 미로를 해결하는것에 그치지 않고 모든 방법론에 적용할 수 있다는 점에 의미가있다. 예를들어, 이 프로젝트를 통해 작성한 알고리즘은 바둑, 오목, 체스와 같은 게임에도 적용가능할 뿐 아니라, 더 나아가서 실제 비디오게임을 플레이하게 할 수도 있다. 그리고 1대1로 경쟁을 하는 게임뿐 아니라 여러 유저를 필요로 하는 게임에 적용시킬 경우 '사람같은' 강화학습 에이전트와 함께 멀티플레이 게임을 즐길 수 있을것이다.

## 2. 환경설정

필자는 다음과 같은 환경에서 작업했다.

> Python - 3.9.12 <br/>
> pygame - 2.1.2 <br/>
> numpy - 1.19.2 <br/>
> torch - 1.7.1 <br/>
> torchvision - 0.8.0a0 <br/>
> matplolib - 3.5.1 <br/>

그리고 미로 코드는 다음 주소의 프로젝트 파일을 수정하여 사용했다.

https://www.pygame.org/project/5609

## 3. Maze_Solver.py 수정

**OpenAI gym**과 같이 작동하는 **MazeSolverEnv**클래스를 만들었다.

> 이는 Maze_Solver.py에 작성되었다.

## 4. REINFORCE (Monte-Carlo Policy-Gradient Method)

![](https://img-blog.csdnimg.cn/f93b0b5423004395b6ad4212806759cf.png)

*출처: Richard Sutton and Andrew Barto. Reinforcement Learning: An Introduction. MIT Press, 1998.*

### 1) Policy Based Gradient Method

위의 의사코드를 보면 알 수 있듯이 Policy인 $ \pi (\cdot | \cdot), \theta) $의 $ \theta $만을 수정하고 있음을 알 수 있다.

**이같이 Policy의(Policy Based) Weight만을 수정(Gradient Method)하는 방법을 Policy Based Gradient Method라 한다.**

### 2) Monte-Carlo Method

**Episode단위로 Policy 또는 Action Value를 수정하는 방법을 Monte-Carlo Method라 한다.**

### 3) Objective Function (목적함수)

**Gradient Method는 목적함수로의 Gradient Ascent이다.**
이는 알고리즘의 다음 부분을 보면 알 수 있다.

$$ \theta \leftarrow \theta + \alpha \gamma^t \nabla ln\, \pi(A_t | S_t, \theta) $$

> 최적화 알고리즘인 Gradient Descent는 다음과 같은 구조를 가진다.
> $\theta \leftarrow \theta - \text{learning rate} \times \text{gradient of Error}$
> 하지만, 강화학습에서 사용하는 Gradient Ascent는 다음과 같다.
> $\theta \leftarrow \theta + \text{learning rate} \times \text{gradient of Objective Function}$

REINFORCEMENT에서의 목적함수 $J_1(\theta)$는 다음과 같다.

$$
J_1(\theta) = V^{\pi_\theta} (s_1) = \mathbb{E}_{\pi_\theta} [v_1]\\
= \mathbb{E}_{\pi_\theta} [G_t | S_t = s_1] = \mathbb{E} \left[ \displaystyle\sum_{k = 0}^{T} \gamma^k R_{t+k+1} | S_t = s_1 \right]\\
 = \displaystyle\sum_{a} \pi(a|s_1, \theta) \displaystyle\sum_{s, r} p(s, r| s_1, a) [r + \gamma v_\pi (s)]
$$

* $s_1$ : Episode 시작시의 State
* $\gamma$ : step-size
* $G(t)$ : Return, $ G(t) = \displaystyle\sum_{k=0}^{\infty} \gamma^{k}R_{t+k+1}$
* $p(s, r| s_1, a)$ : State $s_1$에서 Action $a$를 했을 때, State $s$로 가고 Reward $r$을 얻을 확률

**즉, episodic task에서 REINFORCEMENT의 목적함수는 에피소드를 시작하고 끝날 때까지 얻는 보상의 합이다.**

### 4) 적용

## 5. Actor-Critic Method

![](https://firebasestorage.googleapis.com/v0/b/aing-biology.appspot.com/o/sutton_barto_reinforcement_learning%2Fchapter13%2F02.PNG?alt=media&token=19f2ffc4-aaac-45b2-b537-eb7c02231abd)

*출처: Richard Sutton and Andrew Barto. Reinforcement Learning: An Introduction. MIT Press, 1998.*

### 1) Value Based Method

Policy는 상태(state)들의 집합에서 행동의 확률분포(distribution)로 가는 함수이다.

그리고 상태 $s$에서 Policy $\pi$를 따라가는 State Value Function은 다음과 같이 정의한다.

$ v_\pi (s) = \mathbb{E}{\pi} \left[ G_{t} | S_{t} = s \right] = \mathbb{E}{\pi} \left[ \displaystyle\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_{t} = s \right] \, \text{ for all } s \in \mathcal{S} $

또한, 상태 $ s $에서 행동 $ a $를 선택하고 Policy $ \pi $를 따라가는 Action Value Function은 다음과 같다.

$q_\pi (s, a) = \mathbb{E}{\pi} \left[ G_{t} | S_{t} = s, A_{t} = a \right] = \mathbb{E}{\pi} \left[ \displaystyle\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_{t} = s, A_{t} = a \right] \ \ \text{ for all } s \in \mathcal{S}, a \in \mathcal{A}(s)$

**위같은 함수들을 Value Function이라 하고, 에이전트가 Value Function 기반으로 행동을 결정할 경우 Value Based Method라 한다.**

### 2) Actor-Critic의 의미

위의 의사코드를 보면 알 수 있듯이 Actor-Critic Method는 두 가중치 $ w, \theta $를 업데이트한다.

**이같이, Policy(Actor)와 Value Function(Critic)을 동시에 학습하는 방법을 Actor-Critic Method라 한다.**

> 보면 $ \hat{v} $과 같이 표현했는데, (hat)은 추정값을 의미한다. 그리고 추정값을 사용하는 이유는 환경에서 임의의 상태에 대해 그 Value를 알 수 없기 때문이다.

### 3) Critic과 Variance

위에서 이미 설명했듯이, Policy는 상태에서 행동의 확률로 가는 함수이다.따라서 만약 행동 $a_{1}$이 행동 $a_{2}$보다 '바람직하다'면 $a_{1}$의 확률이 더 높게 나올것이다.

> 여기서 '바람직하다'는 큰 return값을 얻을 수 있음을 의미한다.

이같이, Policy를 업데이트 하는데는 각 행동에 대한 상대적인 수정이 필요하다.

**즉, REINFORCE 알고리즘과 큰 Variance를 가지는 값($G_{t}$)으로 업데이트할 필요가 없다.**

> 큰 값으로 업데이트를 진행하면 Variance가 커지는 이유는 다음 예제를 보면 알 수 있다.

![](https://miro.medium.com/max/1400/1*3r6GvYe9Xm0xWrmNIoatzw.png)

*출처: [Jerry Liu’s post to “Why does the policy gradient method have high variance”](https://www.quora.com/unanswered/Why-does-the-policy-gradient-method-have-a-high-variance)*

이제 $ G_{t} $의 평균값 $ \bar{G} $을 알고있다고 가정하자. 이때, $ G_{t} - \bar{G} $로 업데이트를 하면 Variance를 줄일 수 있다.

**이같이, Variance를 줄이기 위해 사용하는 $ \bar{G} $와 같은 항을 baseline이라 한다.**

이러한 관점에서 Actor-Critic 의사코드에서 $\ G_{t} $대신에 $\ \delta $와 같은 표현을 사용한 이유는 다음과 같다.

---

#### Equation 1

$R_{t+1} + \gamma \hat{v} (s_{t+1}, w) $

$= R_{t+1} + \gamma \mathbb{E} _{\pi} [G_{t+1} | s_{t+1}] $

$= R_{t+1} + \gamma \mathbb{E}_{\pi} \left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+2} | s_{t+1} \right] $

$\approx R_{t+1} + \gamma \left( R_{t+2} + R_{t+2} + \cdots \right) $

$= G_t $

---

#### Equation 2

$ R_{t+1} + \gamma \hat{v} (s_{t+1}, w) $

$ = \hat{q} (s_t, a_t, w) $

---

#### Equation 3

$ \delta = \hat{q} (s_t, a_t, w) - \hat{v} (s_t, w) $

$ = \hat{q} (s_t, a_t, w) - \displaystyle\sum_a \pi(a | s_t) \hat{q} (s_t, a, w) $

* $\pi (a, | s_{t}) $ : 상태 $s_t $에서 행동 $a $를 선택할 확률

---

**Equation 1은 $G_t$대신에 $R_{t+1} + \gamma \hat{v} (s_{t+1}, w)$를 사용할 수 있는 이유를 설명한다.**

**Equation 2는 $R_{t+1} + \gamma \hat{v} (s_{t+1}, w)$와 Action Value Function이 같음을 설명한다.**

**Equation 3는 $\bar{G}$ 대신에 $\hat{v} (s_t, w)$를 사용할 수 있는 이유를 설명한다.**

**이렇게 만들어진 $\delta$를 Advantage라 한다.**

## 6. Deep Q-learning (DQN)

![](https://firebasestorage.googleapis.com/v0/b/aing-biology.appspot.com/o/2022_RL_Lecture%2F27.png?alt=media&token=2d2b0465-984f-40d3-8582-6b8bf3391731)

*출처: Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., and Riedmiller, M. (Dec 2013). Playing Atari with deep reinforcement learning. Technical Report arXiv:1312.5602 [cs.LG], Deepmind Technologies.*

### 1) DQN은 Value Based Method이다.

**DQN의 의사코드를 보면 Policy가 따로 없고 $a_t = \underset{a}{\mathrm{argmax}}\, Q^{*} (\phi(s_t), a; \theta)$와 같은 방법으로 행동을 선택한다.**

위에서 언급했듯이 이는 Value Based Method이다.

### 2) DQN은 Objective Function의 Gradient Ascent가 아닌 Loss의 Gradient Descent이다.

REINFORCEMENT 알고리즘과 Actor-Critic Method의 다음과 같은 업데이트 방식은 Policy의 수렴을 보장한다.

1. REINFORCEMENT

$\theta \leftarrow \theta + \alpha \gamma^t \nabla_\theta ln\,\pi (A_t | S_t, \theta)$

2. Actor-Critic

$\theta \leftarrow \theta + \alpha^\theta I \delta \nabla ln\,\pi (A | S, \theta)$

**하지만 DQN은 Q learning 방식의 TD error를 사용해 Gradient Descent 한다.**

---

#### Q-learning 업데이트 방식

$Q(s_t, a_t) \leftarrow \text{ old Q value } + \text{ TD error}$

$\iff Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma \underset{a}{\mathrm{max}}\, Q (s_{t+1}, a) - Q(s_t, a_t) \right]$

#### DQN의 업데이트 방식

$Q(s_t, a_t; \theta) \leftarrow \text{ old Q value } - \alpha \nabla_\theta \text{ TD error}$

$\iff Q(s_t, a_t; \theta ) \leftarrow Q(s_t, a_t; \theta) + \alpha \nabla_\theta \left[ R_{t+1} + \gamma \underset{a}{\mathrm{max}}\, Q (s_{t+1}, a; \theta) - Q(s_t, a_t; \theta) \right]$

---

### 3) 적용

## 7. 미로 환경 설정

### 1) Preprocessing

#### 1. ANN의 경우 [에이전트 위치, 도착지점의 위치 ,각 타일의 정보]와 같이 인덱싱했다.

맵의 정보는 길을 0 벽을 1로 인덱싱했다. 그리고 만약 시작위치가 [1, 1], 도착지가 [5, 6]이면 state는 다음과 같다.

state = [1, 1, 5, 6, ...,  맵정보]

#### 2. CNN Preprocessing 의사코드는 다음과 같다.

1. 맵 크기만큼의 array에 길은 0, 벽은 1, 이미 가본 길에는 2, 에이전트는 3, 도착지점은 4로 인덱싱했다.
2. 각 인덱스(0, 1, 2, 3, 4)가 하나의 채널을 가지도록 5채널 데이터로 변환한다.

### 2) Reward 시스템

* 각 스텝에 주어진 보상: -0.04
* 벽에 부딪히는 경우: -0.75
* 이미 가본 길을 다시 가는 경우: -0.25
* 도착한 경우: +1

## 8. 발견된 문제점들

### 1) 여전히 Variance가 너무 크다.

![](https://firebasestorage.googleapis.com/v0/b/aing-biology.appspot.com/o/2022_maze_solver_rl%2F01.png?alt=media&token=85946ee6-6201-4f9d-9d06-529d1617567f)
*OpenAI Gym에 Actor-Critic을 적용시킨 경우*

### 2) local optimum

![](https://firebasestorage.googleapis.com/v0/b/aing-biology.appspot.com/o/2022_maze_solver_rl%2F03.png?alt=media&token=82530d1c-b767-4aaf-b503-7923f5994a27)
*미로 탈출 에이전트 학습중*

![](https://firebasestorage.googleapis.com/v0/b/aing-biology.appspot.com/o/2022_maze_solver_rl%2F04.png?alt=media&token=22ad52cb-abef-4c3d-ba2a-a454e1995652)
*미로 에이전트 학습 종료*

위의 첫 번째 사진을 보면 오른쪽 끝 길에서 위 아래로만 움직이고 있다.

**즉, 에이전트는 도착지점(+1)로 가는 것이 아니라 벽과의 충돌을 최소화 하고있다.**

이 증거로 이미 가본 길을 계속 다시 갈 경우 1000 스텝에 Return -250 얻는데, Episode 3000 근처의 return을 보면 -300에 가까운 값을 가지고 Episode 10000 근처에서는 -290에 가까운 값을 가진다. 그리고 실제로 에이전트도 위 아래 움직임만을 반복한다.
