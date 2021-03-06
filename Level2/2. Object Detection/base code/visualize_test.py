import pandas as pd
import json

'''
inference 결과로 만든 csv를 pred_df 변수에 할당하고,
이를 COCO format에 맞는 json 타입으로 변경해줍니다.
'''

# pred_df = pd.read_csv("./ensemble_dir/submission.csv")

pred_df = pd.read_csv("/opt/ml/detection/ensemble_output/submission.csv")
# baseline/faster_rcnn/faster_rcnn_torchvision_submission.csv
# pred_df = pd.read_csv("first_submission.csv")


dic = {
    "info": {},
	"licenses": [],
	"images": [],               # 이미지 list
	"categories": [],           # class 종류들의 list
	"annotations": []           # annotation list
}

dic["info"] = {
    "year": 2021,
    "version": "1.0",
    "description": "Recycle Trash",
    "contributor": "Upstage",
    "url": None,
    "date_created": "2021-02-02 01:10:00"
  }

dic["licenses"] = [
    {
      "id": 0,
      "name": "CC BY 4.0",
      "url": "https://creativecommons.org/licenses/by/4.0/deed.ast"
    }
  ]

dic["categories"] = [
    {
      "id": 0,
      "name": "General trash",
      "supercategory": "General trash"
    },
    {
      "id": 1,
      "name": "Paper",
      "supercategory": "Paper"
    },
    {
      "id": 2,
      "name": "Paper pack",
      "supercategory": "Paper pack"
    },
    {
      "id": 3,
      "name": "Metal",
      "supercategory": "Metal"
    },
    {
      "id": 4,
      "name": "Glass",
      "supercategory": "Glass"
    },
    {
      "id": 5,
      "name": "Plastic",
      "supercategory": "Plastic"
    },
    {
      "id": 6,
      "name": "Styrofoam",
      "supercategory": "Styrofoam"
    },
    {
      "id": 7,
      "name": "Plastic bag",
      "supercategory": "Plastic bag"
    },
    {
      "id": 8,
      "name": "Battery",
      "supercategory": "Battery"
    },
    {
      "id": 9,
      "name": "Clothing",
      "supercategory": "Clothing"
    }
  ]

file_names = pred_df['image_id'].values.tolist()
bboxes = pred_df['PredictionString'].values.tolist()

for f in file_names:
    dic["images"].append({
		"width": 1024,
        "height": 1024,
        "file_name": f,
        "license": 0,
        "flickr_url": None,
        "coco_url": None,
        "id": int(f[5:9])
	})


box_id = 0
for img_idx,box in enumerate(bboxes):
    lst = []
    if type(box) == float:
        continue
    for i,b in enumerate(box.split(" ")):
        if i%6 == 0 and i != 0:
            dic["annotations"].append({
                "image_id": img_idx,          # 어느 이미지에 있는가?
                "category_id": int(lst[0]),   # 이미지의 종류
                "score": float(lst[1]),
                "area": (float(lst[4]) - float(lst[2])) * (float(lst[5]) - float(lst[3])),
                "bbox": [                     # 박스의 좌표
                    float(lst[2]),
                    float(lst[3]),
                    float(lst[4]) - float(lst[2]),
                    float(lst[5]) - float(lst[3])
                ],
                "iscrowd": 0,            # ??
                "id": box_id             # 박스의 ID: 1,2,3, ...
            })
            lst = []
            box_id += 1
        lst.append(b)


with open('csv_to_json.json','w') as f:
    json.dump(dic,f)



# import fiftyone as fo
# import argparse


# def main(arg):
#     dataset = fo.Dataset.from_dir(
#         dataset_type=fo.types.COCODetectionDataset,
#         data_path=arg.data_dir,
#         labels_path=arg.anno_dir,
#     )
#     session = fo.launch_app(dataset, port=arg.port, address="0.0.0.0")
#     session.wait()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', '-d', type=str, default='/opt/ml/detection/dataset',
#                         help='imageData directory')
#     parser.add_argument('--anno_dir', '-a', type=str, default='/opt/ml/detection/csv_to_json.json',
#                         help='annotation Data directory')
#     parser.add_argument('--port', '-p', type=int, default=2223,
#                         help='Port Number')
#     args = parser.parse_args()
#     main(args)