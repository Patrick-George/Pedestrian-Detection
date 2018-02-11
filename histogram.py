import cv2
import numpy as np
import os

sobel_operator_X = np.array([[-1,0,1],[-2,0,2],[-1,0,1]] , dtype=float)
sobel_operator_Y = np.array([[-1,-2,-1],[0,0,0],[+1,+2,+1]]  , dtype=float)
blockSize = 8

def divideInRatio(a,px,b, mag):
    return [abs((px-a)/(b-a))*mag , abs((px-b)/(b-a))*mag]

def corFilt2D(img ,fltMatrix):
    k, _ = fltMatrix.shape
    r,c = img.shape
    k = k//2
    fltImg = np.zeros(img.shape).astype(float)
    for i in range(k, r-1-k):
        for j in range(k,c-1-k):
            roi = img[i-k:i+k+1 , j-k:j+k+1].copy()
            roi = roi.astype(float)
            dst = np.multiply(roi , fltMatrix)
            s = np.sum(dst)
            fltImg[i,j] = s
    return fltImg


def sobelx(img, M=sobel_operator_X):
    sX = corFilt2D(img,M).astype(float)
    sX = abs(sX)
#    sX = np.divide(sX , np.sum(abs(M)))
    return sX

def sobely(img, M=sobel_operator_Y):
    sY = corFilt2D(img,M).astype(float)
    sY = abs(sY)
#    sY = np.divide(sY, np.sum(abs(M)))
    return sY

def sobel(img):
    sY = sobely(img)
    sX = sobelx(img)
    dirn = np.degrees(np.arctan(np.divide(sY , sX))).astype(np.uint8)
    S = np.sqrt(sY**2 + sX**2)/8
    return [S.astype(np.uint8) , dirn ]


def HOGfeatures(img):
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    return sobel(img)

def blockNorm(block):
    return np.divide(block , np.sqrt(np.sum(np.multiply(block, block))))

def bins(dirnBlock , magBlock):
    global blockSize
#    print(dirnBlock.shape)
#    print(magBlock.shape)
    idx = 0
    bin_array = np.zeros((1,9))
    for i in range(blockSize):
        for j in range(blockSize):
            a= (dirnBlock[i,j]//20)*20
            b= a+20
            m,n = divideInRatio(a,dirnBlock[i,j],b,magBlock[i,j])
            bin_array[0,int(a//20)]+=m
            bin_array[0,int(b//20)]+=n
    bin_array = blockNorm(bin_array)
    return list(bin_array.flat)


def operationOverload(filename):
    filename = "pedestrians128x64/"+filename
    im = cv2.imread(str(filename))
    mag, dirn = HOGfeatures(im)

    binBlocks = np.zeros((128,9))
    i=j=k=0
    while i <128:
        while j<64:
            binBlocks[k,:] =np.array(bins(dirn[i:i+8,j:j+8] , mag[i:i+8,j:j+8]))
            k+=1
            j+=8
        i+=8

    binBlocks = list(binBlocks.flat)
    return binBlocks

fout = open("data.txt" , 'w')
directory = "pedestrians128x64"
for filename in os.listdir(directory):
    print(filename)
    binBlocks = operationOverload(filename)
    fout.write(str(binBlocks))
    fout.write('\n')

fout.close()