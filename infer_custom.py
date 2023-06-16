from inference_utils import inference_pipeline

inference_pipeline(model_ckpt_path="C:/Users/kadir/dev/ultralytics/best.pt",
         device="cuda",
         source="test_files/test_video.mp4",
         confidence_threshold=0.25,
         nms_iou_threshold=0.7)