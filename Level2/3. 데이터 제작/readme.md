## 데이터 제작 Competition Wrap Up (4/14 ~ 4/21)

### 1. 프로젝트 목표
- OCR, Text Detection 분야에 대한 이해를 높이기
- Pull Request, Branch를 적극적으로 사용해서 협업 능력을 올리기

### 2. 내가 시도한 점
#### Data
- AI Hub의 야외 글자 데이터를 대회 Annotation 형식에 맞게 변환하는 코드를 작성했다. AI Stage에서 UFO라는 독자적인 OCR Annotation을 사용하고 있어서, 외부 데이터셋을 이에 맞게 변환시켜줘야 함.
- ICDAR2017, ICDAR2019, AI Hub 등.. 출처가 다양한 데이터들을 조합하면서 학습을 시킬 수 있도록 Baseline Code의 train 부분을 수정했다. 
- 오피스아워에서 제공해준 ComposedTransformation 클래스와 Albumentation을 사용하던 기존의 코드들을 Dataset.py에서 입맛에 맞게 사용할 수 있도록 refactoring 했다.
- Multi Scale Training을 통해서 이미지 사이즈를 Epoch마다 변환을 시킴으로써 모델이 같은 이미지라도 다양한 박스를 학습할 수 있게 만들었다.

#### 협업
- 

### 3. 느낀점
#### 마주한 한계와 아쉬운 점 


#### 이전 P-Stage와 비교해서 달라진 점


#### 다음 P-Stage에서 시도할 것 

