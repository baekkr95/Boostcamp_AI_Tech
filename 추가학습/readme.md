## CV 관련 TIPS
### 이미지 시각화
1. [이미지를 opencv로 불러서 matplotlib으로 시각화](https://deep-learning-study.tistory.com/100)  
    - cv2.imread는 BGR로 불러오므로 cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)로 RGB로 색상정보를 변경해야 합니다.

2. [이미지를 Pillow로 불러서 matplotlib으로 시각화](https://kimtaeuk0103.tistory.com/28)
    - PIL로 이미지를 부르고 numpy 타입으로 형변환 후에 plt.imshow() 사용
