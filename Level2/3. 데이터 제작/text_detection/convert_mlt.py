import json
import os
import os.path as osp
from glob import glob
from PIL import Image

import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, ConcatDataset, Dataset


SRC_DATASET_DIR = '/opt/ml/input/data/ICDAR17_CV'  # MLT : All data (9000장), CV : cross validation (7200 + 1800)
DST_DATASET_DIR = '/opt/ml/input/data/ICDAR17_CV'  # MLT : All data (9000장), CV : cross validation (7200 + 1800)

NUM_WORKERS = 32  # FIXME

IMAGE_EXTENSIONS = {'.gif', '.jpg', '.png'}

LANGUAGE_MAP = {
    'Korean': 'ko',
    'Latin': 'en',
    'Symbols': None
}

def get_language_token(x):
    return LANGUAGE_MAP.get(x, 'others')


def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x, exist_ok=True)



class MLT17Dataset(Dataset):
    def __init__(self, image_dir, label_dir, copy_images_to=None):
        image_paths = {x for x in glob(osp.join(image_dir, '*')) if osp.splitext(x)[1] in
                       IMAGE_EXTENSIONS}
        label_paths = set(glob(osp.join(label_dir, '*.txt')))
        assert len(image_paths) == len(label_paths)

        sample_ids, samples_info = list(), dict()
        for image_path in image_paths:
            sample_id = osp.splitext(osp.basename(image_path))[0]

            label_path = osp.join(label_dir, 'gt_{}.txt'.format(sample_id))
            assert label_path in label_paths

            words_info, extra_info = self.parse_label_file(label_path)
            # if 'ko' not in extra_info['languages'] or extra_info['languages'].difference({'ko', 'en'}):
            #     continue

            sample_ids.append(sample_id)
            samples_info[sample_id] = dict(image_path=image_path, label_path=label_path,
                                           words_info=words_info)

        self.sample_ids, self.samples_info = sample_ids, samples_info   # sample_ids: 이미지 이름, sample_info: 이미지의 annotation 정보

        self.copy_images_to = copy_images_to    # copy_images_to : 이미지를 어디에 저장할지

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_info = self.samples_info[self.sample_ids[idx]]

        image_fname = osp.basename(sample_info['image_path'])
        image = Image.open(sample_info['image_path'])
        img_w, img_h = image.size

        if self.copy_images_to:
            maybe_mkdir(self.copy_images_to)
            image.save(osp.join(self.copy_images_to, osp.basename(sample_info['image_path'])))

        license_tag = dict(usability=True, public=True, commercial=True, type='CC-BY-SA',
                           holder=None)
        sample_info_ufo = dict(img_h=img_h, img_w=img_w, words=sample_info['words_info'], tags=None,
                               license_tag=license_tag)

        return image_fname, sample_info_ufo

    def parse_label_file(self, label_path):
        def rearrange_points(points):
            start_idx = np.argmin([np.linalg.norm(p, ord=1) for p in points])   # x와 y를 sum한 뒤에, 가장 작은 값에 해당하는 인덱스 정보를 가져온다. -> 이게 start_idx(좌상단)
            if start_idx != 0:      # 만약 가장 작은 값이 0번째가 아니면 수정을 한다.
                points = np.roll(points, -start_idx, axis=0).tolist() # 제일 작은 값을 맨 앞으로 오게 roll을 함
            return points

        with open(label_path, encoding='utf-8') as f:   # 파일을 개별로 읽어옴
            lines = f.readlines()

        words_info, languages = dict(), set()       # 우리가 원하는 것은 words_info와 어떤 언어인지 알려주는 languages 2가지
        for word_idx, line in enumerate(lines):     # words_info를 얻기 위해 text를 line by line로 읽어온다. -> UFO format에 맞게 만든다.
            items = line.strip().split(',', 9)
            language, transcription = items[8], items[9]
            points = np.array(items[:8], dtype=np.float32).reshape(4, 2).tolist()   # UFO 형식에 맞게 nested list로 만든다
            points = rearrange_points(points)   # x_0, y_0가 무조건 좌상단의 점이여야 한다. -> 이것을 체크하고 수정하는 함수가 rearrage_points

            illegibility = transcription == '###'   # True or False
            orientation = 'Horizontal'
            language = get_language_token(language)
            words_info[word_idx] = dict(
                points=points, transcription=transcription, language=[language],
                illegibility=illegibility, orientation=orientation, word_tags=None
            )
            languages.add(language)

        return words_info, dict(languages=languages)


def main():
    dst_image_dir = osp.join(DST_DATASET_DIR, 'images') # 이미지 디렉토리
    # dst_image_dir = None

    mlt_train = MLT17Dataset(osp.join(SRC_DATASET_DIR, 'raw/ch8_training_images'),
                             osp.join(SRC_DATASET_DIR, 'raw/ch8_training_gt'),
                             copy_images_to=dst_image_dir)
    mlt_valid = MLT17Dataset(osp.join(SRC_DATASET_DIR, 'raw/ch8_validation_images'),
                             osp.join(SRC_DATASET_DIR, 'raw/ch8_validation_gt'),
                             copy_images_to=dst_image_dir)

    # mlt_merged = ConcatDataset([mlt_train, mlt_valid])

    # anno = dict(images=dict())
    # with tqdm(total=len(mlt_merged)) as pbar:
    #     for batch in DataLoader(mlt_merged, num_workers=NUM_WORKERS, collate_fn=lambda x: x):
    #         image_fname, sample_info = batch[0]
    #         anno['images'][image_fname] = sample_info
    #         pbar.update(1)

    # ufo_dir = osp.join(DST_DATASET_DIR, 'ufo')
    # maybe_mkdir(ufo_dir)
    # with open(osp.join(ufo_dir, 'train.json'), 'w') as f:   # annotation 내용을 원하는 디렉토리에 저장
    #     json.dump(anno, f, indent=4)


    ## CV 만들기 : train
    anno = dict(images=dict())
    with tqdm(total=len(mlt_train)) as pbar:
        for batch in DataLoader(mlt_train, num_workers=NUM_WORKERS, collate_fn=lambda x: x):
            image_fname, sample_info = batch[0]
            anno['images'][image_fname] = sample_info
            pbar.update(1)

    ufo_dir = osp.join(DST_DATASET_DIR, 'train_ufo')
    maybe_mkdir(ufo_dir)
    with open(osp.join(ufo_dir, 'train.json'), 'w') as f:   # annotation 내용을 원하는 디렉토리에 저장
        json.dump(anno, f, indent=4)


    ## CV 만들기 : valid
    anno = dict(images=dict())
    with tqdm(total=len(mlt_valid)) as pbar:
        for batch in DataLoader(mlt_valid, num_workers=NUM_WORKERS, collate_fn=lambda x: x):
            image_fname, sample_info = batch[0]
            anno['images'][image_fname] = sample_info
            pbar.update(1)

    ufo_dir = osp.join(DST_DATASET_DIR, 'valid_ufo')
    maybe_mkdir(ufo_dir)
    with open(osp.join(ufo_dir, 'valid.json'), 'w') as f:   # annotation 내용을 원하는 디렉토리에 저장
        json.dump(anno, f, indent=4)


if __name__ == '__main__':
    main()
