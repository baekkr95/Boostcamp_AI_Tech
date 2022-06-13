## Level3 최종 프로젝트 Wrap up

### 1. 프로젝트 목표
- 미세먼지로 인해 배경이 뿌연 이미지를 Dehazing을 하고 하늘과 구름 부분을 맑고 밝은 사진으로 합성해서 사용자에게 제공해주는 서비스를 만드는 것
- 팀에서 Product Serving 부분을 맡아서 이에 대한 지식과 실무적인 능력을 향상시키려고 했다.

### 2. 시도한 점
#### 리서치
- Dehazing Task에 알맞는 Public Dataset을 리서치했고, 각 데이터셋 별 갖고 있는 특징과 링크를 팀 노션에 공유했다.
- Public Dataset은 Pre-trained, Finetuning 단계에서 사용했고, 실제 Testing 단계에서 사용할 이미지를 구글에서 크롤링하거나 주변 지인들에게서 사진을 모아 사용했다.
- 갖고 있는 데이터로 실제 프로젝트를 진행할 수 있는지 확인해보기 위해, 여러 Dehazing Model에 테스팅용 데이터를 Inference 해봤고 이를 바탕으로 모델을 개선시킬 수 있었다.

#### 프로토타입
- Streamlit을 사용해서 Front-End를 구성했다.
- Dehazing Model, Sky Replacement Model이 추가될 때마다 UI와 기능을 바꾸면서 프로토타입 버전을 올렸다.

#### 프로덕트 서빙
- Back-End 통신으로 FastAPI를 사용했다.
- Dehazing, Segmentation, Selection, Replacement, Save 총 5가지 API 기능을 만들었다.
- DB로는 MongoDB를 사용했고, 합성에 사용될 구름 정보와 사용자가 서비스를 사용하면서 생성한 정보들을 저장했다.


### 3. 아쉬운 점, 개선 점
- Front-End에 Streamlit을 사용하다보니 UI 구성이 너무 단순했고, 앱 서비스화를 할 수 없었다.
- GCP, Docker를 프로젝트에 적용해보질 못했다.
- 
