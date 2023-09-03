import cv2
import numpy as np
import matplotlib.pyplot as plt


def barFill(bar):
    proportion = np.count_nonzero(bar)/bar.shape[0]
    window = bar.shape[0]//50
    ret = np.zeros(bar.shape)
    if(window==0):
        return ret
    sum = 0
    for i in range(len(bar)-1):
        botLim = np.maximum(i-window, 0)
        upLim = np.minimum(i+window, len(bar))
        if botLim > 0:
            sum -= bar[botLim-1]
        if upLim < len(bar):
            sum += bar[upLim]
        ret[i] = (sum/window) > proportion
    return ret

def lightAdjustment(img):
    img = img.copy()
    blured = cv2.blur(img, [50,50])
    cv2.imwrite('blur.jpg', blured)

    adjust_strength = 0.79
    avg = np.mean(np.mean(blured))
    map = adjust_strength * avg / blured + (1-adjust_strength)
    cv2.imwrite('map.jpg', map*256)

    img = np.uint8(np.multiply(img, map))
    img = cv2.equalizeHist(img)
    cv2.imwrite('img.jpg', img)

    return img

def cutEdgeTopBot(img):
    img = img.copy()
    th = 255 * img.shape[1]//16
    summed = np.sum(img, axis=1)
    top = 0
    bot = -1
    for i in range(len(summed)-1):
        if(summed[i]<=th):
            top = i
            break
    for i in range(len(summed)-1,0,-1):
        if(summed[i]<=th):
            bot = i
            break
    if(top==bot):
        return []
    return img[top:bot, :], top, bot

def lineScan(img):
    origin = img.copy()
    #img = cv2.equalizeHist(img)
    img = lightAdjustment(img)

    #cv2.imshow('equalizeHist',img)
    #cv2.waitKey(5000)

    ret, img = cv2.threshold(img,232,256,cv2.THRESH_BINARY)

    img = cv2.medianBlur(img, 3)

    kernel = np.ones((1, img.shape[1]//10), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((3, 1), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    cv2.imwrite('threshold.jpg', img)
    #cv2.waitKey(2000)

    img, top, bot = cutEdgeTopBot(img)
    origin = origin[top:bot, :]
    #cv2.imwrite("strange.jpg", img)
    if(img.shape[0]<1):
        print("Error: 切除上下边缘时发生错误")
        return

    #cv2.imshow('edgecutted',img)
    #cv2.imwrite('filled.jpg', img)
    #cv2.waitKey(5000)

    bar = np.sum(img, axis=1)
    avg = np.sum(bar)/(img.shape[0] * 2.0)
    bar = (bar - avg)>0

    #plt.bar(range(img.shape[0]), bar)
    #plt.show()

    segged = barFill(bar)

    #plt.bar(range(segged.shape[0]), segged)
    #plt.show()

    ret = []
    i = 0
    j = 1
    while(j<segged.shape[0] and i<segged.shape[0]-1):
        leftEdge = not segged[i] and segged[i+1]
        rightEdge = segged[j-1] and not segged[j]
        j = max(j, i+1)
        if(leftEdge and rightEdge):
            ret.append((i+j)/2)
            i+=1; j+=1
            continue
        if(not rightEdge):
            j+=1
        if(not leftEdge):
            i+=1
    #plt.bar(range(img.shape[0]), segged)
    #plt.show()

    #index = 0
    #temp = origin[ret[index][0]:ret[index][1],:]
    #cv2.imshow('', temp)
    #cv2.waitKey(0)

    #print(np.count_nonzero(ret))
    imgsMaskPairs = []
    for i, row in enumerate(ret):
        pair = []
        if i==0:
            lbound = 0
        else:
            lbound = (int)((row+ret[i-1])/2)

        if i==(len(ret)-1):
            hbound = -1
        else:
            hbound = (int)((row+ret[i+1])/2)
        pair.append(origin[lbound:hbound, :])
        pair.append(img[lbound:hbound, :])
        imgsMaskPairs.append(pair)

    return imgsMaskPairs

def getTextAndArrowFromPairs(imgsMaskPairs):
    textArrowPairs = []
    for img, mask in imgsMaskPairs:
        imgArea = img.shape[0] * img.shape[1]
        imgHeight= img.shape[0]
        imgWidth  = img.shape[1]
        pair = []
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if(len(contours) > 0):
            x,y,w,h = cv2.boundingRect(contours[0])
            if(w < imgWidth/16 or h < imgHeight/8):
                continue
            text = img[y:y+h,x:x+w]
            pair.append(text)
        else:
            pair.append(None)

        if(len(contours) > 1):
            x,y,w,h = cv2.boundingRect(contours[1])
            arrow = img[y:y+h,x:x+w]
            #print('w:' + str(w) + ', h:' + str(h))
            #cv2.imshow('arrow', arrow)
            #cv2.waitKey(1000)
            if(w < imgWidth/32 or h < imgHeight/8):
                pair.append(None)
                textArrowPairs.append(pair)
                #print("twosmall")
                continue
            elif(w/h > 4.0 or h/w > 4.0):
                pair.append(None)
            else:
                pair.append(arrow)
        else:
            pair.append(None)
        textArrowPairs.append(pair)

    return textArrowPairs




if __name__ == "__main__":
    img = cv2.imread("D:/workspace/Python/engPic/lineScan04.jpg")[:,:,1]
    imgs = lineScan(img)
    i = 0
    for img in imgs:
        i += 1
        cv2.imwrite("output/" + str(i) + ".jpg", img)










#not used below
#not used below
#not used below
def floodEdge(img):
    shape = img.shape
    seeds = np.array([[0,0], [shape[0]-1, 0], [0, shape[1]-1], [shape[0]-1, shape[1]-1]])
    #mask = np.zeros([img.shape[0]+2, img.shape[1]+2], dtype=np.uint8)
    for seed in seeds:
        print(seed)
        retval, img, m, rect = cv2.floodFill(image=img, mask=None, seedPoint=seed, newVal=0)
        print(img)
        print()
    img = cv2.medianBlur(img, 3)
    return img