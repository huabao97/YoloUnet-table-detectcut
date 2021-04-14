#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
table detect with yolo 
@author: chineseocr
"""
import cv2
import numpy as np
from config import tableModelDetectPath
from utils import nms_box,letterbox_image,rectangle

tableDetectNet  = cv2.dnn.readNetFromDarknet(tableModelDetectPath.replace('.weights','.cfg'),tableModelDetectPath)#

def table_detect(img,sc=(416,416),thresh=0.5,NMSthresh=0.3):
    """
    表格检测
    img:GBR

    """
    scale =sc[0]
    img_height,img_width = img.shape[:2]
    inputBlob,fx,fy = letterbox_image(img[...,::-1],(scale,scale))
    inputBlob = cv2.dnn.blobFromImage(inputBlob, scalefactor=1.0, size=(scale,scale),swapRB=True ,crop=False);
    tableDetectNet.setInput(inputBlob/255.0)
    outputName = tableDetectNet.getUnconnectedOutLayersNames()
    outputs = tableDetectNet.forward(outputName)
    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > thresh:
                    center_x = int(detection[0] * scale/fx)
                    center_y = int(detection[1] * scale/fy)
                    width = int(detection[2] * scale/fx)
                    height = int(detection[3] * scale/fy)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    if class_id==1:
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        xmin ,ymin,xmax,ymax= left,top,left+width, top+height
                        xmin = max(xmin,1)
                        ymin = max(ymin,1)
                        xmax = min(xmax,img_width-1)
                        ymax = min(ymax,img_height-1)
                        boxes.append([xmin,ymin,xmax,ymax ])
        
    boxes = np.array(boxes)
    
    confidences = np.array(confidences)
    if len(boxes)>0:
        boxes, confidences =nms_box(boxes, confidences, score_threshold=thresh, nms_threshold=NMSthresh)
        

        
    boxes,adBoxes=fix_table_box_for_table_line(boxes,confidences,img)
    return boxes,adBoxes,confidences

def point_in_box(p,box):
    x,y = p
    xmin,ymin,xmax,ymax = box
    if xmin<=x<=xmin and ymin<=y<=ymax:
        return True
    else:
        return False
    
      

def fix_table_box_for_table_line(boxes,confidences,img):
    ### 修正表格用于表格线检测 
    h,w = img.shape[:2]
    n =len(boxes)
    adBoxes = []
    
    for i in range(n):
        prob = confidences[i]
        
        xmin,ymin,xmax,ymax = boxes[i]
        padx = (xmax-xmin)*(1-prob)
        padx = padx
        
        pady = (ymax-ymin)*(1-prob)
        pady = pady
        xminNew = max(xmin-padx,1)
        yminNew = max(ymin-pady,1)
        xmaxNew = min(xmax+padx,w)
        ymaxNew = min(ymax+pady,h)
        
         
        adBoxes.append([xminNew,yminNew,xmaxNew,ymaxNew])   
        
                
    return boxes,adBoxes
        
        
        
    

if __name__=='__main__':
    import time
    import os
    import cv2
    # 源目录
    MyPath = 'D:/table/table-detect-master/test_in/'
    # 输出目录
    OutPath = 'D:/table/table-detect-master/test_out/'
    #剪切目录
    imgcutpath = 'D:/table/table-detect-master/imgcut/'
    os.chdir(MyPath)
    for i in os.listdir(os.getcwd()):
        postfix =i
       #saveimg=i
        postfix = cv2.imread(postfix)
        # print(type(postfix))
        t = time.time()
        boxes, adBoxes, scores = table_detect(postfix, sc=(416, 416), thresh=0.5, NMSthresh=0.3)
        # print(scores)
        print("正在处理" + i + "图片")
        print("处理时间",time.time() - t, "边框坐标",boxes,"回归边框坐标",adBoxes,"置信度",scores)
        img = rectangle(postfix, adBoxes)
        #print(os.getcwd())
        img.save(OutPath+ i,"png")
        #cv2.imwrite(OutPath + i+ ".png", postfix)
        #print(MyPath + i)
        #saveimg = cv2.imread(MyPath + i)
        #print(type(saveimg))
        os.chdir(OutPath)
        dirpath=str(OutPath+i)
        # print(dirpath)
        img1=cv2.imread(dirpath)
        # print(img1)
        # cv2.imshow("img1",img1)
        # cv2.waitKey(0)
        a = 'table'  # 类别名
        font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
        for k in range(len(adBoxes)):
            b = scores[k]
            img1= cv2.putText(img1, '{} {:.3f}'.format(a, b), (int(adBoxes[k][0]), int(adBoxes[k][1])), font,2 ,(0, 255, 0),5)
        cv2.imwrite(dirpath,img1)
        img = cv2.imread(MyPath+i)
        # print(type(img))
        for k in range(len(adBoxes)):
            print("剪切第" + str(k) + "表格区域")
            # print(len(adBoxes))
            crop1 = img[int(adBoxes[k][1]):int(adBoxes[k][3]),int(adBoxes[k][0]):int(adBoxes[k][2])]# 裁剪坐标为[y0:y1, x0:x1]
            cv2.imwrite(imgcutpath+str(k)+i,crop1)
            print("剪切成功")

#
    # p = 'img/3det.png'
    # img = cv2.imread(p)
    # t =time.time()
    # boxes,adBoxes,scores=table_detect(img, sc=(416, 416),thresh=0.5,NMSthresh=0.3)
    # print(time.time()-t,boxes,adBoxes,scores)
    # img = rectangle(img,adBoxes)
    # img.save('img/3dettable.png')
    # import cv2
    # saveimg = cv2.imread("img/3dettable.png")
    # print(saveimg.shape)
    # imgcutpath="imgcut/"
    # for i in range(len(adBoxes)):
    #     cropped = saveimg[int(adBoxes[i][1]):int(adBoxes[i][3]), int(adBoxes[i][0]):int(adBoxes[i][2])]  # 裁剪坐标为[y0:y1, x0:x1]
    #     cv2.imwrite(imgcutpath + str(i) + ".png", cropped)