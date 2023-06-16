import torch
import numpy as np
import cv2
import json
from pathlib import Path
from glob import glob
from ultralytics.yolo.utils import DEFAULT_CFG, DEFAULT_CFG_DICT,ops
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.augment import LetterBox
import logging

# init console logger
LOGGER = logging.getLogger('YOLOv8')
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

def load_model_from_ckpt(model_ckpt_path, device):
    """
    Load the detection model into torch device and configure for the inference.
    Args:
        model_checkpoint_path: str
            YOLO model checkpoint file local path
        device: torch device where the model loaded to. "cpu" or "cuda"
    Returns:
        (model, cfg): tuple
            Loaded model and configuration
    """

    model_path = str(Path(model_ckpt_path))
    if device =="cuda":
        if torch.cuda.is_available():
            device = "cuda" 
        else:
            device = "cpu"
            LOGGER.warning('Cuda device is not available on this device or environment. Changed to cpu inference.')

    device = torch.device(device)
    torch_model = torch.load(model_path,map_location=device)

    args = {**DEFAULT_CFG_DICT, **(torch_model.get('train_args', {}))}  # combine model and default args, preferring model args
    model = (torch_model.get('ema') or torch_model['model']).to(device).float()  # FP32 model
    model.args = args  # attach args to model
    model.pt_path = model_path  # attach *.pt file path to model
    model.task = 'detect'
    model = model.eval() # set model to eval mode to cancel gradient calculation during inference
    model = model.to(device)
    model = model.fuse(verbose=False)
    stride = max(int(model.stride.max()), 32)
    model.names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    model.half() if args["half"] else model.float()
    model.eval()
    model.pt = True
    model.fp16 = False
    model.stride = stride
    cfg = get_cfg(DEFAULT_CFG,args)
    LOGGER.info('Model loaded from checkpoint:  %s',model_ckpt_path)
    return model, cfg


def pre_transform(model,imgsz,im):
    """Pre-tranform input image before inference. Add padding for incompatible width-height ratio images.
    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
    Returns: A list of transformed imgs.
    """
    same_shapes = all(x.shape == im[0].shape for x in im)
    auto = same_shapes and model.pt
    return [LetterBox(imgsz, auto=auto, stride=model.stride)(image=x) for x in im]

def preprocess(model, imgsz, im, device):
    """Prepare input image before inference. Load to desired torch device.
    Args:
        model (nn.tasks.DetectionModel): YOLOv8 detection model  
        im (torch.Tensor | List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
        imgsz (Tuple(int,int)): Model input size
        device (torch.device): pytorch cuda or cpu device

    Returns:
        im (torch.Tensor): Preprocessed image as torch tensor.
    """
    if not isinstance(im, torch.Tensor):
        im = np.stack(pre_transform(model, imgsz, im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)
    # NOTE: assuming im with (b, 3, h, w) if it's a tensor
    img = im.to(device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    return img

def postprocess(args,model, preds, img, orig_imgs):
    """
    Postprocesses predictions and returns a list of result dicts.
    Eliminate bbox proposals that have conf score lower than conf threshold 
    and IOU with other proposals higher than NMS IOU threshold
    Args:
        args: Detection model arguments
        model (nn.tasks.DetectionModel): YOLOv8 detection model  
        preds (Tuple(torch.Tensor, List(torch.Tensor))): YOLO model output, unfiltered bbox proposals
        img(torch.Tensor): Preprocessed image as torch tensor.
        orig_imgs: Original image as List (h, w, 3)
    Returns : 
        List[Dict]: List of Dict of bboxes with conf score and class name.
    """

    preds = ops.non_max_suppression(preds,
                                    args.conf,
                                    args.iou,
                                    agnostic=args.agnostic_nms,
                                    max_det=args.max_det,
                                    classes=args.classes)
    results = []
    for i, pred in enumerate(preds[:1]): # currently inference script works properly for only single image, batch is not implemented
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        if not isinstance(orig_imgs, torch.Tensor):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape) # Scale the output wrt orig. img size
        for p in pred:
            result = {}
            box = p[:4]
            conf = p[-2]
            class_name = model.names[int(p[-1])]
            result["bbox"] = box.int().tolist()
            result["conf"] = float(conf)
            result["class_name"] = class_name
            results.append(result)
    return results

def plot_bboxes_labels_and_scores_on_frame(frame, results):
    """
    Args:
        frame: original image as List. (h, w, 3)
        results: List[Dict]: List of Dict of bboxes with conf score and class name.
    Returns:
        plotted_img: Image as list (h, w, 3), with bboxes and class names plotted on orig img 
    """
    plotted_img = frame  # return original image if no pred exists
    for result in results:
        x1y1 = result["bbox"][:2]
        x2y2 = result["bbox"][2:]
        text = result["class_name"]+" "+("{0:.2f}".format((result["conf"])))
        plotted_img = cv2.rectangle(img=frame, pt1=x1y1, pt2=x2y2, color=(255, 0, 255), thickness=1)
        plotted_img = cv2.putText(img=plotted_img, text=text, org=x1y1,fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=1, thickness=1, lineType=cv2.LINE_AA,color=(255, 0, 255))
    return plotted_img

def show(p, plotted_img):
    """Display an image in a window using OpenCV imshow()."""
    cv2.imshow(str(p)+" - Hold down any key to stop inference", plotted_img)
    cv2.waitKey(1)  # 1 millisecond

def frame_generator(source):
    """
    Generate a video stream from webcam or local video, get single frame in each iteration.

    Args:
        source (str): Video source, 0 or "0" for webcam,  video path for local video. 
    Returns:
        image: Generated next frame as List(h, w, 3) 
    """
    out_path = generate_result_out_path(source)

    source = 0 if source == "0" else source
    video_capture = cv2.VideoCapture(source)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    size = (width, height)
    try:
        while cv2.waitKey(1) == -1:
            ret, frame = video_capture.read()
            if not ret:
                break
            yield frame, out_path, fps, size
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        raise StopIteration
    finally:
        video_capture.release()

def read_valid_img(file):
    """
    Check image extension and read and return if it's a supported format.
    Args:
        file: Image file path
    Returns:
        image: Image that read from local path as List(h, w, 3) 
    """
    if any(file.endswith(ext) for ext in [".jpeg",".jpg",".png"]):
        img = cv2.imread(file)
        return img
    raise ValueError(f"Unsupported image source: {file}")

def img_folder_iterator(source):
    """
    Generate a image folder iterator, get a single image that supported format in each iteration.
    Args:
        source (str): Image folder path 
    Returns:
        image: Generated next frame as List(h, w, 3) 
    """
    path = str(Path(source) / "*")
    files = glob(path)
    for file in files:
        img =  read_valid_img(file)
        out_path = generate_result_out_path(file)
        yield img, out_path, None, None  # None values are used to utilize pipeline's generic for loop
                                         # and are dummy values equivalent to frame_generator function's 
                                         # fps and img_size variables

def generate_result_out_path(source):
    """
    Check file whether is a supported format, if supported, generate the output file's name.
    Args:
        source (str): image path, video path or "0" (for webcam video)
    Returns:
        output_name (str): Output filename, that with prefix "pred_out_*"  
    """
    if str(source) == "0":
        path = Path().cwd() / "pred_out_webcam.mp4"
    elif Path(source).is_file() and Path(source).suffix in [".jpeg",".jpg",".png",".avi",".mp4"]:
        path = Path(source).parent / str("pred_out_"+Path(source).name)
    elif not Path(source).is_dir():
        raise ValueError(f"Invalid input source: '{source}'")
    else:  
        path = Path(source)
    return str(path)


def inference_pipeline(model_ckpt_path,device,source, confidence_threshold,nms_iou_threshold):
    """ Perform inference on YOLOv8 model. Show and save prediction results as json.
    Args:
        model_ckpt_path (str): YOLOv8 model checkpoint local path
        device: Desired torch device to load the model, "cuda" or "cpu"
        source: Image path, video path, image folder path or "0" (for webcam) to perform inference on
        confidence_threshold: Model confidence threshold. Predictions with conf score lower than this threshold will  be eliminated
        nms_iou_threshold: Postprocess NMS IOU threshold. Predictions that overlapping with other preds with higher IOU value than this
                        threshold will be eliminated during postprocess
    """

    model, cfg = load_model_from_ckpt(model_ckpt_path=model_ckpt_path,device=device)
    cfg.conf = confidence_threshold
    cfg.iou = nms_iou_threshold

    result_json_path = "result.json"
    is_source_image_folder = False
    video_writer = None
    result_json = {}
    result_json["image_to_preds"] = {}
    count = 0
    
    if str(source) == "0" or any(str(source).endswith(ext) for ext in [".avi",".mp4"]):
        result_json["source"] = "webcam" if str(source) == "0" else source
        image_loader = frame_generator(str(source))
    elif Path(source).is_file(): # generate single element image iterator for the below loop
        result_json["source"] = source
        image_loader = iter([(read_valid_img(source),generate_result_out_path(source),None ,None)])
    elif Path(source).is_dir():  # generate image list iterator for img dir
        is_source_image_folder = True
        result_json["source"] = source
        image_loader = img_folder_iterator(source)
    else: raise ValueError(f"Unsupported input source {source}")

    for frame, out_path, fps, frame_size in image_loader:
        if fps is not None and (video_writer is None):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
        im0s = [frame]

        im = preprocess(model,[cfg.imgsz,cfg.imgsz], im0s, device)
        preds = model(im)
        results = postprocess(cfg, model, preds, im, im0s)
        
        plotted_img = plot_bboxes_labels_and_scores_on_frame(frame,results)
        if not is_source_image_folder:
            show(out_path,plotted_img)
        if video_writer:
            video_writer.write(plotted_img)
            result_json["image_to_preds"][count] = results
        else:
            cv2.imwrite(out_path,plotted_img)
            result_json["image_to_preds"][out_path] = results   
        count +=1
        if cv2.waitKey(1) != -1:
            break
    if video_writer:
        video_writer.release()
    else:  # source is not a video
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    with open(result_json_path, "w") as fs:
        json.dump(result_json,fs,indent=1)