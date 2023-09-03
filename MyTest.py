import Inference
import torch
import cv2
import argparse
import arrownet.linescan as ls
#import arrownet.ArrowNet as an
from arrownet.ResNet18 import ResNet18
import numpy as np
from textoutput import Textproc
from arrownet.TextProcessor import TextProcessor
from arrownet.ArrowNet import SignBotData
#from ultralytics import YOLO
#from utils.tools import *
#import ast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./database/IMG_20230628_095351.jpg", help="path to image file"
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
        "--output", type=str, default="./output/", help="image save path"
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
    parser.add_argument(
        "--box_prompt", type=str, default="[0,0,0,0]", help="[x,y,w,h]"
    )
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def imgsshow(imgs):
    for i, img in enumerate(imgs):
        cv2.imshow(str(i), img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

def pairshow(pairs):
    for firstImg, secondImg in pairs:
        cv2.imshow('1', firstImg)
        if secondImg is not None:
            cv2.imshow('2', secondImg)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    DIM = (224, 224)
    lineImgProc = Textproc()
    path = './weights/ArrowNet'
    arrowModel = ResNet18(img_channels = 1, num_classes = 4).to(device)
    arrowModel.load_state_dict(torch.load(path))
    tp = TextProcessor()
    args = parse_args()

    path = "./database/"
    csv = "test_data.csv"
    dataset = SignBotData(path, csv, device)

    correctCount = 0
    cantFindError = 0
    misjudgeError = 0

    for i in range(dataset.n_sample):
        print(i)
        path = dataset.x[i]
        target = dataset.t[i]
        correctDir = dataset.d[i]
    
        args = parse_args()
        args.img_path = path
        corpedImgs = Inference.main(args)
        imgsMaskPairs = []
        #imgsshow(corpedImgs)
        #cv2.imwrite('croped.jpg', corpedImgs[0])
        #cv2.imwrite('croped_green.jpg', corpedImgs[0][:,:,1])
        for img in corpedImgs:
            imgsMaskPairs = imgsMaskPairs + ls.lineScan(img[:,:,1])
        #imgsMaskPairs = imgsMaskPairs + ls.lineScan(corpedImgs[:,:,1])
        #print('got \''+ str(len(imgsMaskPairs)) +'\' imgsMaskPairs')
        #pairshow(imgsMaskPairs)
        #cv2.imwrite('rowimg.jpg', imgsMaskPairs[1][0])
        #cv2.imwrite('rowmask.jpg', imgsMaskPairs[1][1])
        textArrowPairs = ls.getTextAndArrowFromPairs(imgsMaskPairs)
        #print('got \''+ str(len(textArrowPairs)) +'\' textArrowPairs')
        #pairshow(textArrowPairs)
        #cv2.imwrite('text.jpg', textArrowPairs[1][0])
        #cv2.imwrite('arrow.jpg', textArrowPairs[1][1])

        tp.clearLines()
        for text, arrow in textArrowPairs:
            label, confidence = lineImgProc.readImg(text, gray=True)
            #print(label)
            #print(confidence)
            if arrow is None:
                tp.addLineWithoutArr(label[0])
            else:
                #cv2.imshow('', arrow)
                #cv2.waitKey(2500)
                arrow = cv2.resize(arrow, DIM)/256
                arrow = np.float32(arrow.reshape(1, 1, 224, 224))
                arrow = torch.tensor(arrow).to(device)
                dir = arrowModel(arrow).cpu().detach().numpy()
                #print(dir)
                tp.addLine(label[0], dir)
            cv2.destroyAllWindows()
        #print(tp)

        rl = tp.searchForLoc(target)
        if rl is not None:
            if(rl.arrowDir==correctDir):
                print("Correct!" + " Turn to direction: " + str(rl.arrowDir))
                correctCount = correctCount + 1
            else:
                print("Error! direction error" + " Turn to direction: " + str(rl.arrowDir))
                print("But we should" + " Turn to direction: " + str(correctDir))
                misjudgeError = misjudgeError + 1
        else:
            print("Error! Not find the room!")
            cantFindError = cantFindError + 1
    print(correctCount)
    print(misjudgeError)
    print(cantFindError)
    #img = cv2.imread("./output/4.jpg")


    #imgsshow(imgs)

    #textProc.outputImgs(imgs)
    #textProc.readImg(img, gray=False)
    #textProc.readImgs(imgs, gray=True)
    