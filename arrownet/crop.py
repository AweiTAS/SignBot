import numpy as np
import cv2
import sympy

def appx_best_fit_ngon(contour, n: int = 4) -> list[(int, int)]:
    hull = cv2.convexHull(contour)
    hull = np.array(hull).reshape((len(hull), 2))
    # to sympy land
    hull = [sympy.Point(*pt) for pt in hull]
    # run until we cut down to n vertices
    while len(hull) > n:
        best_candidate = None

        # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # we need to first make sure that the sum of the interior angles the edge
            # makes with the two adjacent edges is more than 180°
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

            # find the new vertex if we delete this edge
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]

            # the area of the triangle we'll be adding
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
            # should be the lowest
            if best_candidate and best_candidate[1] < area:
                continue

            # delete the edge and add the intersection of adjacent edges to the hull
            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)

        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]

    # back to python land
    hull = [(int(x), int(y)) for x, y in hull]

    return hull


def largest_4gon_in_mask(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours)==0):
        return
    largestIndex = 0
    maxArea = cv2.contourArea(contours[largestIndex])
    for i in range(1, len(contours)):
        area = cv2.contourArea(contours[i])
        if(area > maxArea):
            maxArea = area
            largestIndex = i
    box = appx_best_fit_ngon(contours[largestIndex])
    box = np.int0(box)
    return box

def find_largest_4gon_in_ann(ann):
    ann = sorted(ann, reverse=True, key=lambda an:np.sum(np.sum(an, axis=0), axis=0))
    ann = ann[0:4]
    ret = []
    for an in ann:
        mask = cv2.Mat(np.uint8(an))
        cnt = largest_4gon_in_mask(mask)
        ret.append(cnt)
    return ret

def find_largest_4gon_in_mask(mask):
    return largest_4gon_in_mask(mask)

def bottomLeft_arrange(cnt):
    axisX = 0
    axisY = 1

    if(cnt[0][axisX] <= cnt[1][axisX]):
        minX = 0
        secX = 1
    else:
        minX = 1
        secX = 0

    for i in range(2,4):
        if(cnt[i][axisX] <= cnt[minX][axisX]):
            secX = minX
            minX = i
        elif(cnt[i][axisX] <= cnt[secX][axisX]):
            secX = i

    if(cnt[minX][axisY] <= cnt[secX][axisY]):
        index = minX
    else:
        index = secX

    return np.roll(cnt, -index*2)
'''
def crop_From_Cnt(img, cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    box = bottomLeft_arrange(cnt)
    
    regularSize_bottomLeft = np.array([[0,0],[w,0],[w,h],[0,h]])
    H, _ = cv2.findHomography(box, regularSize_bottomLeft)
    warped = cv2.warpPerspective(img, H, (w, h))
    img = cv2.drawContours(img, [box], -1, (0,255,0), 3)

    return img, warped
'''
def crop_From_Cnt(img, cnts):
    imgForDraw = img.copy()
    retImg = []
    retWarped = []
    for i, cnt in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(cnt)
        box = bottomLeft_arrange(cnt)
        
        regularSize_bottomLeft = np.array([[0,0],[w,0],[w,h],[0,h]])
        H, _ = cv2.findHomography(box, regularSize_bottomLeft)
        retWarped.append(np.uint8(cv2.warpPerspective(img, H, (w, h))))
        cv2.imwrite("temp"+str(i)+".jpg", np.uint8(cv2.warpPerspective(img, H, (1920, 1080))))
        retImg.append(cv2.drawContours(imgForDraw, [box], -1, (0,255,0), 3))

    return retImg, retWarped