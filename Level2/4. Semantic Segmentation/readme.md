## Segmentation Competition 개인 Wrap up (4/25 ~ 5/12)

### 1. 프로젝트 목표
- Git Flow를 참고해서 브랜치와 PR 전략을 세우고 이를 최대한 활용해서 협업 능력을 향상시키는 것이 제일 큰 목표였다.
- 저번 대회(Data-Centric)를 참고삼아 데이터 관점에서 성능을 향상시키려고 했다.

### 2. 내가 시도한 점
#### Data
- Train과 Test Dataset을 검수해보면서 각 클래스별로 주요 특징을 파악해봤다.
- 각 클래스별 특징을 고려해서 적절한 Augmentation 기법을 적용해봤고, 하나의 모델을 기준으로 비교 분석해봤다.
- 

#### Model
- 모델의 다양성 측면을 고려해서 인코더에 다양한 모델을 사용해봤다.
- mmseg 라이브러리에서 Transformer 계열이 많이 사용됐기 때문에, Torch SMP 라이브러리에 있는 Convolution 계열의 네트워크를 사용했다.
- 


### 3. 느낀 점
#### 아쉬운 점


#### 이전 P-Stage와 달라진 점


