import cv2
import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule

class Textproc:

    parseq = None
    img_transform = None
    def __init__(self) -> None:
        self.parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
        self.img_transform = SceneTextDataModule.get_transform(self.parseq.hparams.img_size)

    def imgPreprocess(self, img):
        # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
        img = self.img_transform(img).unsqueeze(0)
        logits = self.parseq(img)
        # Greedy decoding
        pred = logits.softmax(-1)
        return pred

    def outputImgs(self, imgs):
        for i, img in enumerate(imgs):
            cv2.imwrite("output/" + str(i) + ".jpg", img)

    def readImg(self, img, gray=True):
        if(gray):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pred = self.imgPreprocess(Image.fromarray(img))
        label, confidence = self.parseq.tokenizer.decode(pred)
        return label, confidence
    
    def readImgs(self, imgs, gray=True):
        ret = []
        for i, img in enumerate(imgs):
            label, confidence = self.readImg(img, gray)
            ret.append([label, confidence])