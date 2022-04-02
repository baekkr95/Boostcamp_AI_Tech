import fiftyone as fo
import argparse


def main(arg):
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=arg.data_dir,
        labels_path=arg.anno_dir,
    )
    session = fo.launch_app(dataset, port=arg.port, address="0.0.0.0")
    session.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default='/opt/ml/detection/dataset',
                        help='imageData directory')
    parser.add_argument('--anno_dir', '-a', type=str, default='/opt/ml/detection/csv_to_json.json',
                        help='annotation Data directory')
    parser.add_argument('--port', '-p', type=int, default=2223,
                        help='Port Number')
    args = parser.parse_args()
    main(args)

# usage
# cd detection
# python visualize.py

# FO를 사용해서 데이터를 볼 수 있습니다.