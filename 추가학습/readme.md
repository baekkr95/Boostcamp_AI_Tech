## 파이토치 및 CV TIPS
### 이미지 처리
1. [이미지를 opencv로 불러서 matplotlib으로 시각화](https://deep-learning-study.tistory.com/100)  
    - cv2.imread는 BGR로 불러오므로 cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)로 RGB로 색상정보를 변경해야 합니다.
    - 시각화에서 BGR로 출력이 된다면, 이미지 변수 뒤에 [:,:,::-1]를 하면 RGB로 바뀌어서 출력이 된다.

2. [이미지를 Pillow로 불러서 matplotlib으로 시각화](https://kimtaeuk0103.tistory.com/28)
    - PIL로 이미지를 부르고 numpy 타입으로 형변환 후에 plt.imshow() 사용


3. [tensor 이미지를 matplotlib으로 시각화](https://ndb796.tistory.com/373?category=1011147)
    - PyTorch의 경우 [Batch Size, Channel, Width, Height]의 구조를 가지고 있어서, 이를 matplotlib로 출력하기 위해서는 [Width, Height, Channel]의 순서로 변경해주어야 한다.

4. [Albumentations 공식 문서](https://albumentations.ai/docs/getting_started/installation/)
    - 기본적으로 opencv가 필요하다. 
    - 참고 블로그 : https://albumentations.ai/docs/getting_started/installation/

5. [cutmix 구현, 공식문서](https://github.com/clovaai/CutMix-PyTorch/tree/2d8eb68faff7fe4962776ad51d175c3b01a25734)
    - Loss function은 하나, cutmix를 쓸때 안 쓸때 구분만 하면 된다.

6. Visualizing augmentated images
    - 원하는 augmentation이 잘 적용되고 있는지 확인해보는 과정은 필수다.

### 함수
1. [Detach와 Clone 차이점](https://subinium.github.io/pytorch-Tensor-Variable/)
    - 둘 다 기존 Tensor를 복사하는 함수
    - `detach()` : 기존 Tensor에서 gradient 전파가 안되는 텐서 생성한다. 단, storage를 공유하기에 detach로 생성한 Tensor가 변경되면 원본 Tensor도 똑같이 변합니다.
    - `clone()` : 기존 Tensor와 내용을 복사한 텐서 생성

2. Contiguous


### Loss
1. 불균형 데이터 관련
    - Focal Loss
    - Label Smoothing : https://ratsgo.github.io/insight-notes/docs/interpretable/smoothing
    - Weighted Sampler
    - Imbalanced dataset sampler (oversampling)
    - 
