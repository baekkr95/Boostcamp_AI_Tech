## CV 관련 TIPS
### 이미지 시각화
1. [이미지를 opencv로 불러서 matplotlib으로 시각화](https://deep-learning-study.tistory.com/100)  
    - cv2.imread는 BGR로 불러오므로 cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)로 RGB로 색상정보를 변경해야 합니다.

2. [이미지를 Pillow로 불러서 matplotlib으로 시각화](https://kimtaeuk0103.tistory.com/28)
    - PIL로 이미지를 부르고 numpy 타입으로 형변환 후에 plt.imshow() 사용


3. [tensor 이미지를 matplotlib으로 시각화](https://ndb796.tistory.com/373?category=1011147)
    - PyTorch의 경우 [Batch Size, Channel, Width, Height]의 구조를 가지고 있어서, 이를 matplotlib로 출력하기 위해서는 [Width, Height, Channel]의 순서로 변경해주어야 한다.

4. [Albumentations 공식 문서](https://albumentations.ai/docs/getting_started/installation/)
    - 기본적으로 opencv가 필요하다. 
    - 참고 블로그 : https://albumentations.ai/docs/getting_started/installation/


### Loss
1. 불균형 데이터, Focal Loss
