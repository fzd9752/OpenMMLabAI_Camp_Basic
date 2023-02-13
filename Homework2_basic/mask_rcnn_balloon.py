_base_ = ['mask_rcnn_r50_fpn_2x_coco.py',]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
# dataset_type = 'COCODataset'
classes = ('balloon',)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        img_prefix='balloon_dataset/balloon/train',
        classes=classes,
        ann_file='balloon_dataset/balloon/train/train.json'),
    val=dict(
        img_prefix='balloon_dataset/balloon/val/',
        classes=classes,
        ann_file='balloon_dataset/balloon/val/val.json'),
    test=dict(
        img_prefix='balloon_dataset/balloon/val/',
        classes=classes,
        ann_file='balloon_dataset/balloon/val/val.json')
)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'configs/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'

optimizer = dict(type='SGD', lr=0.0015, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=48)
checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
