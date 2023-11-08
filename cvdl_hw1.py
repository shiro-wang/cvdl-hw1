from PyQt5 import QtWidgets
from cvdl_hw1_ui import Ui_MainWindow
import sys
import cv2
import numpy as np
from tkinter.filedialog import askdirectory
from PyQt5.QtWidgets import QFileDialog
import os
from os.path import join

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchsummary import summary
import torchvision.transforms.functional as TF
#import matplotlib.image as mpimg
from PIL import Image

from random import randrange
import matplotlib.pyplot as plt
from VGG19_train import VGG19


class MainWindow(QtWidgets.QMainWindow):
    isCalibrate = False

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


        choices = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        self.ui.comboBox.addItems(choices)

        # ButtonEvents
        # 0. Load Img
        self.ui.pushButton_01.clicked.connect(self.LoadAllImg)
        self.ui.pushButton_02.clicked.connect(self.LoadImgL)
        self.ui.pushButton_03.clicked.connect(self.LoadImgR)

        # 1. Camera Calibration
        self.ui.pushButton_11.clicked.connect(self.CornerDetect)
        self.ui.pushButton_12.clicked.connect(self.FindIntrinsicMat)
        self.ui.pushButton_13.clicked.connect(self.FindExtrinsicMat)
        self.ui.pushButton_14.clicked.connect(self.FindDistortionMat)
        self.ui.pushButton_15.clicked.connect(self.UnDistort)

        # 2. Augmented Reality
        self.ui.pushButton_21.clicked.connect(self.ShowAR)
        self.ui.pushButton_22.clicked.connect(self.ShowARVertical)
        
        # 3. Stereo Disparity Map
        self.ui.pushButton_31.clicked.connect(self.Stereo)

        # 4. SIFT
        self.ui.pushButton_41.clicked.connect(self.LoadImg1)
        self.ui.pushButton_42.clicked.connect(self.LoadImg2)
        self.ui.pushButton_43.clicked.connect(self.Keypoints)
        self.ui.pushButton_44.clicked.connect(self.MatchKeypoints)

        # 5. VGG19
        self.ui.pushButton_50.clicked.connect(self.LoadImg)
        self.ui.pushButton_51.clicked.connect(self.ShowAugmentedImg)
        self.ui.pushButton_52.clicked.connect(self.ShowModel)
        self.ui.pushButton_53.clicked.connect(self.ShowAccuracyLoss)
        self.ui.pushButton_54.clicked.connect(self.Inference)
        
        self.img = {
            "result": {
                "1": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "2": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "3": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "4": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "5": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "6": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "7": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "8": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "9": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "10": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "11": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "12": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "13": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "14": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "15": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp')
            }
        }

        folderPath = " "
        folderLPath = " "
        folderRPath = " "
        folderName = " "
        folderLName = " " 
        folderRName = " "
        fileCount = 0

    def LoadAllImg(self):
        global folderPath
        folderPath = QFileDialog.getExistingDirectory(self, "Open folder")
        global fileCount
        fileCount = 0

        for path in os.listdir(folderPath):
            if os.path.isfile(os.path.join(folderPath, path)) and "bmp" in path:
                fileCount += 1
        print(fileCount)        

    def LoadImgL(self):
        self.filenameL, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.img_L = cv2.imread(self.filenameL)

    def LoadImgR(self):
        self.filenameR, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.img_R = cv2.imread(self.filenameR) 
    
    def CornerDetect(self):
        global fileCount
        global folderPath
        global fileName
        for i in range(1,fileCount + 1):
            fileName = folderPath +"/"+ str(i) + ".bmp"
            tempColor = cv2.imread(fileName, cv2.IMREAD_COLOR)
            tempGray = cv2.imread(fileName, cv2.COLOR_RGB2GRAY)
            self.img["result"][str(i)] = tempGray
            # findChessboardCorners(img, (The number of rows and columns of the corner points inside the checkerboard: w,h \
            # ), the detected corner points(?) )
            ret, corner = cv2.findChessboardCorners(self.img["result"][str(i)], (11, 8), None)
            
            if ret == True:
                # creteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
                # corner2 = cv2.cornerSubPix(self.img["result"][str(i)], corner, (5,5), (-1,-1), creteria)
                cv2.drawChessboardCorners(self.img["result"][str(i)], (11, 8), corner, ret)
                # for count, pt in enumerate(corner2):
                #     cv2.circle(self.img["result"][str(i)], int(pt[0][0]), int(pt[0][1]), 10, (0, int(255/len(corner2)*count), 0))

        for i in range(1,fileCount + 1):
            # cv2.imshow("chessboard corners", cv2.resize(tempColor, (900, 900)))
            cv2.imshow("chessboard corners", cv2.resize(self.img["result"][str(i)], (900, 900)))
            cv2.waitKey(500)
        
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    # 1. 相機標定
    # 目的在於：建立相機成像幾何模型並矯正透視畸變，透過多個視角的2D/3D對應，求解出該相機的內參數與每一個視角的外參數
    # calibrateCaamera所給的世界座標方向，是由objectPoints的設定順序，以及findChessboardCoreners的偵測順序共同決定
    # From 3D to 2D: sm' = A[R|t]M'
    # [R|t]: 相機的外部參數矩陣, turn world coordinates to camera coordinates
    # A: 相機的內參數矩陣, turn camera coordinates to image coordinates
    # mat:A, dist:(畸變矩陣)distortion, rvec:R, tvec:t
    def FindIntrinsicMat(self):
        global fileCount
        global folderPath
        global fileName
        objPoint = np.zeros((11*8, 3), np.float32)
        objPoint[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        # print(objPoint)
        objpoints = []
        imgpoints = []

        for i in range(1, fileCount+1):
            fileName = folderPath +"/"+ str(i) + ".bmp"
            tempImg = cv2.imread(fileName)
            pic = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
            ret, corner = cv2.findChessboardCorners(pic, (11, 8), None)
            if ret == True:
                objpoints.append(objPoint)
                imgpoints.append(corner)

        ret, mat, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (1024, 1024), None, None)
        print("Intrinsic")
        print(mat)

    def FindExtrinsicMat(self):
        global fileCount
        global folderPath
        global fileName
        objp = np.zeros((11*8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        rets = []
        corners = []

        for i in range(1,fileCount + 1):
            fileName = folderPath +"/"+ str(i) + ".bmp"
            tempImg = cv2.imread(fileName)
            pic = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
            ret, corner = cv2.findChessboardCorners(pic, (11, 8), None)
            rets.append(ret)
            corners.append(corner)
            objpoints.append(objp)
            imgpoints.append(corner)
        rets, mat, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (1024, 1024), None, None)
        pic_num = self.ui.comboBox.currentIndex()
        rotation = cv2.Rodrigues(rvec[pic_num])[0]

        print("Extrinsic:")
        print("[[{:f}, {:f}, {:f}, {:f}]".format(rotation[0][0], rotation[0][1], rotation[0][2], tvec[pic_num][0][0]))
        print("[{:f}, {:f}, {:f}, {:f}]".format(rotation[1][0], rotation[1][1], rotation[1][2], tvec[pic_num][1][0]))
        print("[{:f}, {:f}, {:f}, {:f}]]".format(rotation[2][0], rotation[2][1], rotation[2][2], tvec[pic_num][2][0]))
    
    def FindDistortionMat(self):
        global fileCount
        global folderPath
        global fileName
        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        
        for i in range(1, fileCount + 1):
            fileName = folderPath +"/"+ str(i) + ".bmp"
            tempImg = cv2.imread(fileName)
            pic = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
            ret, corner = cv2.findChessboardCorners(pic, (11, 8), None)           
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corner)
                
        ret, mat, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (1024, 1024), None, None)
        with np.printoptions(precision = 6, suppress = True):
            print("Distortion:")
            print("[{}]".format(dist[0]))
    
    def UnDistort(self):
        global fileCount
        global folderPath
        global fileName
        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        
        for i in range(1, fileCount + 1):
            fileName = folderPath +"/"+ str(i) + ".bmp"
            tempImg = cv2.imread(fileName)
            pic = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
            ret, corner = cv2.findChessboardCorners(pic, (11, 8), None)           
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corner)
        ret, mat, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (1024, 1024), None, None)
        for i in range(1, fileCount + 1):
            fileName = folderPath +"/"+ str(i) + ".bmp"
            tempImg = cv2.imread(fileName)
            self.img["result"][str(i)] = cv2.undistort(tempImg,mat,dist,None,mat)
            pic_show = np.hstack([cv2.resize(tempImg, (600, 600)),cv2.resize(self.img["result"][str(i)], (600, 600))])
            cv2.imshow("undistort",pic_show)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

    # 2. Augmented Reality
    def ShowAR(self):
        global fileCount
        global folderPath
        global fileName
        # get library folder
        libraryPath = os.path.join(folderPath, 'Q2_lib', 'alphabet_lib_onboard.txt')
        print(libraryPath)
        fs = cv2.FileStorage(libraryPath,cv2.FILE_STORAGE_READ)

        # get textbox word  
        keyWord = self.ui.textbox.text()

        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        imgs = []

        for i in range(fileCount):
            fileName = folderPath +"/"+ str(i+1) + ".bmp"
            tempImg = cv2.imread(fileName)
            imgs.append(tempImg)
            ret, corners = cv2.findChessboardCorners(imgs[i], (11, 8), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        ret, mat, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (1024, 1024), None, None)
        
        # for each file
        for i in range(fileCount):
            for j in range(len(keyWord)):
                if j == 0:
                    dy = 5
                    dx = 7
                elif j == 1:
                    dy = 5
                    dx = 4           
                elif j == 2:
                    dy = 5
                    dx = 1
                elif j == 3:
                    dy = 2
                    dx = 7
                elif j == 4:
                    dy = 2
                    dx = 4
                elif j == 5:
                    dy = 2
                    dx = 1
                    
                ch = np.float32(fs.getNode(keyWord[j]).mat())    
                for k in range(ch.shape[0]):
                    ch[k][0][0] = ch[k][0][0] + dx
                    ch[k][0][1] = ch[k][0][1] + dy
                    ch[k][1][0] = ch[k][1][0] + dx
                    ch[k][1][1] = ch[k][1][1] + dy
                    ch[k] = np.float32(ch[k])
                    imgpoints,_ = cv2.projectPoints(ch[k],rvec[i],tvec[i],mat,dist)
                    imgs[i] = cv2.line(imgs[i],(int(imgpoints[0][0][0]),int(imgpoints[0][0][1])),(int(imgpoints[1][0][0]),int(imgpoints[1][0][1])), (0, 0, 255), 5)

        # show image
        cv2.namedWindow("AR", cv2.WINDOW_NORMAL)
        for i in range(0, fileCount):
            cv2.resizeWindow("AR", 600, 600)
            cv2.imshow("AR", imgs[i])
            cv2.waitKey(1000)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def ShowARVertical(self):
        global fileCount
        global folderPath
        global fileName
        # get library folder
        libraryPath = join(folderPath, 'Q2_lib', 'alphabet_lib_vertical.txt')
        fs = cv2.FileStorage(libraryPath,cv2.FILE_STORAGE_READ)

        # get textbox word  
        keyWord = self.ui.textbox.text()

        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        imgs = []
        
        for i in range(fileCount):
            fileName = folderPath +"/"+ str(i+1) + ".bmp"
            tempImg = cv2.imread(fileName)
            imgs.append(tempImg)
            ret, corners = cv2.findChessboardCorners(imgs[i], (11, 8), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        ret, mat, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (1024, 1024), None, None)
        
        # for each file
        for i in range(fileCount):
            for j in range(len(keyWord)):
                if j == 0:
                    dy = 5
                    dx = 7
                elif j == 1:
                    dy = 5
                    dx = 4           
                elif j == 2:
                    dy = 5
                    dx = 1
                elif j == 3:
                    dy = 2
                    dx = 7
                elif j == 4:
                    dy = 2
                    dx = 4
                elif j == 5:
                    dy = 2
                    dx = 1

                ch = np.float32(fs.getNode(keyWord[j]).mat())    
                for k in range(ch.shape[0]):
                    ch[k][0][0] = ch[k][0][0] + dx
                    ch[k][0][1] = ch[k][0][1] + dy
                    ch[k][1][0] = ch[k][1][0] + dx
                    ch[k][1][1] = ch[k][1][1] + dy
                    ch[k] = np.float32(ch[k])
                    imgpoints,_ = cv2.projectPoints(ch[k],rvec[i],tvec[i],mat,dist)
                    imgs[i] = cv2.line(imgs[i],(int(imgpoints[0][0][0]),int(imgpoints[0][0][1])),(int(imgpoints[1][0][0]),int(imgpoints[1][0][1])), (0, 0, 255), 5)

        # show image
        cv2.namedWindow("AR", cv2.WINDOW_NORMAL)
        for i in range(0, fileCount):
            cv2.resizeWindow("AR", 600, 600)
            cv2.imshow("AR", imgs[i])
            cv2.waitKey(1000)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    # 3. Stereo Display Map
    def Stereo(self):
        # save original image 
        self.img_L_org = self.img_L
        self.img_R_org = self.img_R
        # image gray
        self.img_L = cv2.cvtColor(self.img_L, cv2.COLOR_BGR2GRAY)
        self.img_R = cv2.cvtColor(self.img_R, cv2.COLOR_BGR2GRAY)
        # find the disparity image
        stereoBM = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        self.disparity = cv2.normalize(stereoBM.compute(self.img_L, self.img_R), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.disparity_org = stereoBM.compute(self.img_L, self.img_R)

        cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
        cv2.imshow("disparity", self.disparity)
        cv2.resizeWindow("disparity", 1000, 680)

        event = cv2.EVENT_LBUTTONDOWN
        cv2.namedWindow('imgL', cv2.WINDOW_NORMAL)
        cv2.namedWindow('imgR_dot', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('imgL', self.drawCircle)
        cv2.resizeWindow("imgL", 1000, 680)
        cv2.resizeWindow("imgR_dot", 1000, 680)
        cv2.imshow('imgL',self.img_L_org)
        cv2.imshow('imgR_dot',self.img_R_org)
        cv2.waitKey(0)

    def Stereo(self):
        # save original image 
        self.img_L_org = self.img_L
        self.img_R_org = self.img_R
        # image gray
        self.img_L = cv2.cvtColor(self.img_L, cv2.COLOR_BGR2GRAY)
        self.img_R = cv2.cvtColor(self.img_R, cv2.COLOR_BGR2GRAY)
        # find the disparity image
        stereoBM = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        self.disparity = cv2.normalize(stereoBM.compute(self.img_L, self.img_R), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.disparity_org = stereoBM.compute(self.img_L, self.img_R)
        event = cv2.EVENT_LBUTTONDOWN
        # self.drawCircle(self, event, x, y, flags, param)

        cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
        cv2.imshow("disparity", self.disparity)
        cv2.resizeWindow("disparity", 1000, 680)

        cv2.namedWindow('imgL', cv2.WINDOW_NORMAL)
        cv2.namedWindow('imgR_dot', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('imgL', self.drawCircle)
        cv2.resizeWindow("imgL", 1000, 680)
        cv2.resizeWindow("imgR_dot", 1000, 680)
        cv2.imshow('imgL', self.img_L_org)
        cv2.imshow('imgR_dot', self.img_R_org)

    def drawCircle(self, event, x, y, flags, param):
            # 追蹤繪製
        drawing = False
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

        if drawing:
            d = self.disparity_org[y][x] / 16
            if d > 0:
                img_R_dot = np.copy(self.img_R_org)
                img_L_dot = np.copy(self.img_L_org)

                cv2.circle(img_R_dot, (int(x - d), y), 10, (0, 255, 0), -1)
                print((int(x - d), y),'dis:',d)
                cv2.circle(img_L_dot, (x, y), 10, (0, 0, 255), -1)
                cv2.imshow('imgL', img_L_dot)
                cv2.imshow('imgR_dot', img_R_dot)
            elif d < 0:
                img_R_dot = np.copy(self.img_R_org)
                img_L_dot = np.copy(self.img_L_org)
                
                cv2.circle(img_L_dot, (x, y), 10, (0, 0, 255), -1)
                cv2.imshow('imgL', img_L_dot)
                cv2.imshow('imgR_dot', img_R_dot)
                print('Failure case')
    
    def LoadImg1(self):
        self.filenameL, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.img_1 = cv2.imread(self.filenameL)

    def LoadImg2(self):
        self.filenameR, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.img_2 = cv2.imread(self.filenameR) 

    def Keypoints(self):
        self.img_1 = cv2.cvtColor(self.img_1, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kps, des = sift.detectAndCompute(self.img_1, None)
        results = cv2.drawKeypoints(self.img_1, kps, (0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.namedWindow("sift_keypoints", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("sift_keypoints", 1000, 680)
        cv2.imshow('sift_keypoints', results)

    def MatchKeypoints(self):
        self.img_1 = cv2.cvtColor(self.img_1, cv2.COLOR_BGR2GRAY)
        self.img_2 = cv2.cvtColor(self.img_2, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kps1, des1 = sift.detectAndCompute(self.img_1, None)
        kps2, des2 = sift.detectAndCompute(self.img_2, None)
        draw1 = cv2.drawKeypoints(self.img_1, kps1, 0, (0,255,0), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        draw2 = cv2.drawKeypoints(self.img_2, kps2, 0, (0,255,0), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        # matches = sorted(matches, key = lambda x:x.distance)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        results = cv2.drawMatchesKnn(draw1,kps1,draw2,kps2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.namedWindow("left", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("left", 1000, 680)
        cv2.imshow('left', draw1)
        cv2.namedWindow("right", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("right", 1000, 680)
        cv2.imshow('right', draw2)
        cv2.namedWindow("match", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("match", 1000, 680)
        cv2.imshow('match', results)

    # 5. VGG19 with BN
    def LoadImg(self):
        self.filename, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.img = Image.open(self.filename).convert("RGB")

    def ShowAugmentedImg(self):
        label = ['automobile', 'bird' ,'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        #(x_train, y_train), (x_test, y_test) = train_dataset.load_data()
        transform_1 = transforms.Compose([transforms.ToTensor(),transforms.RandomRotation(180, expand=True, center=None)])
        transform_2 = transforms.Compose([transforms.ToTensor(),transforms.RandomResizedCrop(size = (200,200),scale=(0.2, 1.0), ratio=(0.5, 1.1))])
        transform_3 = transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip(p=0.9)])
        
        plt.figure(figsize=(8,8),facecolor='w')
        for i in range(3):
          for j in range(3):
            self.fileName = "./Q5_image/Q5_1/"+ str(label[i*3+j]) + ".png"
            img_0 = Image.open(self.fileName).convert("RGB")
            img_1 = transform_1(img_0).permute(1,2,0).numpy()
            img_2 = transform_2(img_1).permute(1,2,0).numpy()
            img_3 = transform_3(img_2).permute(1,2,0).numpy()
            plt.subplot(3, 3, i*3+j+1)
            plt.title(label[i*3+j])
            plt.imshow(img_3)
            plt.axis('off')
        plt.show()

    def ShowModel(self):
        model = VGG19(10)
        MODELS_PATH = './models'
        MODELS_SAVE = 'VGG19_model_1.pth'
        modelPath = os.path.join(MODELS_PATH, MODELS_SAVE)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)
        if device == 'cpu':
            model.load_state_dict(torch.load(f=modelPath, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(f=modelPath, map_location=torch.device('cuda')))
        model.eval()
        summary(model, (3, 32, 32))  

    def ShowAccuracyLoss(self):
        img1 = cv2.imread('acc.png')
        img2 = cv2.imread('loss.png')
        plt.figure(figsize=(11,6))
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.axis('off')
        plt.title('Loss')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.axis('off')
        plt.title('Acc')

        plt.show()

    def Inference(self):
        model = VGG19(10)
        MODELS_PATH = './models'
        MODELS_SAVE = 'VGG19_model_1.pth'
        modelPath = os.path.join(MODELS_PATH, MODELS_SAVE)
        model.load_state_dict(torch.load(f=modelPath, map_location=torch.device('cuda')))
        
        model.eval()
        #image = Image.open('0.jpg')
        x = TF.to_tensor(self.img)
        x.unsqueeze_(0)
        with torch.inference_mode():
            predict = model(x)
            classes = ['airplane', 'automobile', 'bird' ,'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            print(predict[1])
            print(classes[predict[1].argmax()])
            print(predict[1].max().item())

            plt.figure(figsize=(11,6))
            plt.subplot(1, 2, 1)
            plt.imshow(self.img)
            plt.title(f"Confidence = {str(predict[1].max().item())[0:4]}\nPrediction Label = {classes[predict[1].argmax()]}")
            plt.axis('off')
            # plt.show()

            plt.subplot(1, 2, 2)
            plt.title('Probability of each classes')
            plt.bar(classes, predict[1][0])
            plt.xticks(rotation=30)
            plt.xlabel('Classes')
            plt.ylabel('Probability')

            plt.show()
            

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
