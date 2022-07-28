# 사용할 모델 선택
_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'

# 모델 class수 변경
model = dict(
  roi_head = dict(
    bbox_head = dict(
      num_classes = 1
    ),
    mask_head = dict(
      num_classes = 1
    )
  )
)

# 데이터 폴더 설정
data_root = 'data/dataset/'
classes = ('Normal',)

# 데이터 설정
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
      img_prefix=data_root + "train/",
      classes = classes,
      ann_file=data_root + "label(polygon)_train.json"
),
    val=dict(
        img_prefix=data_root + "train/",
        classes = classes,
        ann_file=data_root + "label(polygon)_train.json"
),
    test=dict(
        img_prefix=data_root + "test/",
        classes = classes,
        ann_file=data_root + "test.json"
)
)





# log 저장 위치
checkpoint_config = dict(interval=1,out_dir='work_dirs/lg_mask/')

# 평가 방법
evaluation = dict(interval=1, metric=['bbox', 'segm'])

# 사전 가중치 사용
load_from = 'checkpoint/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

# epoch 설정 
runner = dict(type='EpochBasedRunner', max_epochs=5)

# batch size 설정
auto_scale_lr = dict(enable=False, base_batch_size=16)