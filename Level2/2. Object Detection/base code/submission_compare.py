from map_boxes import mean_average_precision_for_boxes
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO

'''
앙상블 성능 비교 전략
1. Fold 3 train data로 학습해서 모델을 여러개 생성
2. 1번에서 나온 모델들로 앙상블 모델 생성
3. 생성된 앙상블 모델로 Fold 3 valid data를 inference해서 "앙상블.csv" 파일 생성
4. 우리가 이미 갖고 있는 Fold 3 valid json 파일을 사용해서 "앙상블.csv" 파일과 비교를 한 후에 mAP 계산

pip install map_boxes 필수
'''

# valid data의 json 파일
VALID_JSON = '/opt/ml/detection/stratified_kfold/cv_val_3.json'
# 앙상블 모델로 만든 csv 파일
PRED_CSV = '/opt/ml/detection/baseline/mmdetection/tools/submission100.csv'
# classes
LABEL_NAME = ["General trash", "Paper", "Paper pack", "Metal", 
              "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]



# load ground truth(VALID_JSON)
with open(VALID_JSON, 'r') as outfile:
    test_anno = (json.load(outfile))

# load prediction(앙상블 csv파일)
pred_df = pd.read_csv(PRED_CSV)

'''
mAP 계산을 위해 mean_average_precision_for_boxes 함수를 사용해야 한다.

하지만, 우리가 갖고 있는 csv 파일 양식과 달리, 밑에 예시와 같은 양식을 사용하기 때문에
new_pred라는 새로운 리스트를 만들어서 양식을 맞추는 작업이 필요함.

[
    [file_name label_index confidence_score x_min x_max y_min y_max], 
    [file_name label_index confidence_score x_min x_max y_min y_max],
    ,,,
    [file_name label_index confidence_score x_min x_max y_min y_max]
]
'''
# 양식 맞추기
new_pred = []

file_names = pred_df['image_id'].values.tolist()
bboxes = pred_df['PredictionString'].values.tolist()

for i, bbox in enumerate(bboxes):
    if isinstance(bbox, float):
        print(f'{file_names[i]} empty box')

for file_name, bbox in tqdm(zip(file_names, bboxes)):
    boxes = np.array(str(bbox).split(' '))
    
    if len(boxes) % 6 == 1:
        boxes = boxes[:-1].reshape(-1, 6)
    elif len(boxes) % 6 == 0:
        boxes = boxes.reshape(-1, 6)
    else:
        raise Exception('error', 'invalid box count')
    for box in boxes:
        new_pred.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])


'''
이제 기존 valid data의 양식도 gt 라는 리스트를 만들어서
mean_average_precision_for_boxes에 맞게 변형을 시켜줘야 함.

여기서는 confidence score가 필요가 없다.
[
    [file_name 1, label_index, x_min, x_max, y_min, y_max], 
    [file_name 2, label_index, x_min, x_max, y_min, y_max],
    ,,,
    [file_name , label_index, x_min, x_max, y_min, y_max]
]
'''
    
gt = []

coco = COCO(VALID_JSON)
   
for image_id in coco.getImgIds():
        
    image_info = coco.loadImgs(image_id)[0]
    annotation_id = coco.getAnnIds(imgIds=image_info['id'])
    annotation_info_list = coco.loadAnns(annotation_id)
        
    file_name = image_info['file_name']
        
    for annotation in annotation_info_list:
        gt.append([file_name, annotation['category_id'],
                   float(annotation['bbox'][0]),
                   float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                   float(annotation['bbox'][1]),
                   (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])

mean_ap, average_precisions = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=0.5)

print(mean_ap)
for i in test_anno['categories']:
    print(i['id'], ':', i['name'])
