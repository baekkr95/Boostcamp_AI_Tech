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
- Wandb에서 데이터를 Image Logging 한 뒤에 어떤 데이터에 대해서 학습이 잘 안되고 있는지 파악해볼 수 있었다.

![image](https://user-images.githubusercontent.com/48708496/164912064-dac286da-83aa-4055-a681-f20bd05c6565.png)

#### 협업
- 저번 대회에는 아이디어, 작업 현황, 의견 공유 등을 깃허브 Issue 한 곳에 몰아서 작성을 하다보니 너무 번잡하다고 느껴서 공간을 분리하기로 했다.
- 이를 위해 아이디어 공유, 대회와 관련된 레퍼런스, 데이터 분류 테이블 등에 관한 내용은 노션에만 했고, Issue에는 진행 중인 작업과 의견을 올리기로 했다.

![image](https://user-images.githubusercontent.com/48708496/164911289-0ba29e4c-1f59-4b18-87be-d994ee0ee1e0.png)

- 이번에 가장 공을 들여서 협업한 부분은 `PR(Pull Request)`와 `Branch` 전략이었다.
- Main 브랜치에서 팀원 각자 기능 중심의 Branch를 만들었다. 나 같은 경우에는 Data와 Augmentation에 대한 작업만 하는 브랜치를 만들었다.
- 기능 브랜치를 Main에 Merge하기 위해서 PR templates을 통해 해당 PR에 대한 내용을 적을 수 있게 만들었고 팀원 모두의 Code Review와 승인을 받아야만 Merge를 할 수 있도록 했다.

![image](https://user-images.githubusercontent.com/48708496/164911875-641f8f34-3295-416e-92ae-a33879786043.png)


### 3. 느낀점
한줄 느낀점 : 질 좋은 Data의 중요성에 대해 크게 깨닫게 됐다. Data-Centric!

#### 마주한 한계와 아쉬운 점 
- 아직 Git 사용법이 많이 미숙하다는 것을 깨달았다. 프로젝트 중간에 Conflict가 발생했는데, 이에 대한 원인을 파악하고 해결하는데 거의 반나절 이상이 걸렸다.
- EDA, Wandb Image Logging, Sweep setting, Fiftyone, 데이터 수집 등.. 초반 Setup에 너무 시간을 많이 쏟아서 Augmentation 실험을 충분히 하지 못했다.

#### 이전 P-Stage와 비교해서 달라진 점
- Pull Request와 Branch를 사용하면서 Git에 대해 더욱 자신감이 생겼다.
- Wandb Image Logging을 통해 시각화를 부분을 개선했다.
- 아이디어, 레퍼런스, Issue를 분리함으로써 더 효율적으로 협업할 수 있었다.

#### 다음 P-Stage에서 시도할 것 
- 1순위는 Git을 더 많이, 잘 사용하는 것
- Branch name과 Wandb experiment name에 대한 컨벤션을 추가할 생각이다.
- 백그라운드 실행을 사용할 예정이다. (Nohup)
- PR code review을 효율적으로 하는 방법을 찾자.
