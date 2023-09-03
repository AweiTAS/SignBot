import argparse
from fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy
import cv2
import numpy as np
import time
from arrownet import crop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./SegOutput/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()


def main(args):
    # load model
    model = FastSAM(args.model_path)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)
    dim = (1920,1080)
    input = Image.open(args.img_path).resize(dim)
    input = input.convert("RGB")
    everything_results = model(
        input,
        device=args.device,
        retina_masks=args.retina,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou
        )
    bboxes = None
    points = None
    point_label = None
    prompt_process = FastSAMPrompt(input, everything_results, device=args.device)

    ann = prompt_process.color_prompt(RGBcolor=np.array([109,000,157]), maxDis=70, secondMaxDis=75)
    #contours, hierarchy = cv2.findContours(ann[1].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #drawedImg = cv2.drawContours(cv2.resize(cv2.imread(args.img_path), dim, interpolation = cv2.INTER_AREA), contours, -1, (0,255,0), 3)
    #imgtemp = ann[1]*256
    #cv2.imwrite('temp_out_mask.jpg', imgtemp)
    #cv2.imwrite('tempcontour_mask.jpg', drawedImg)
    cnts = crop.find_largest_4gon_in_ann(ann)
    
    drawedImgs, warpeds = crop.crop_From_Cnt(cv2.resize(cv2.imread(args.img_path), dim, interpolation = cv2.INTER_AREA), cnts)
    #cv2.imwrite('tempcontour_mask_qua.jpg', drawedImgs[1])

    return warpeds
    prompt_process.plot(
        annotations=ann,
        output_path=args.output+args.img_path.split("/")[-1],
        bboxes = bboxes,
        points = points,
        point_label = point_label,
        withContours = args.withContours,
        better_quality=args.better_quality,
    )

    #return
    #ann = prompt_process.everything_prompt()
    #return
    


if __name__ == "__main__":
    args = parse_args()
    main(args)
