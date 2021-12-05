import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
import time

start = time.time()

# Grayscale
def BGR2GRAY(img):
	# Grayscale
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray

# Canny Edge dedection
def Canny_edge(img):
	# Canny Edge 
    canny_edges = cv2.Canny(img,100,200)
    return canny_edges

# Gabor Filter
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
	# get half size
	d = K_size // 2

	# prepare kernel
	gabor = np.zeros((K_size, K_size), dtype=np.float32)

	# each value
	for y in range(K_size):
		for x in range(K_size):
			# distance from center
			px = x - d
			py = y - d

			# degree -> radian
			theta = angle / 180. * np.pi

			# get kernel x
			_x = np.cos(theta) * px + np.sin(theta) * py

			# get kernel y
			_y = -np.sin(theta) * px + np.cos(theta) * py

			# fill kernel
			gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)

	# kernel normalization
	gabor /= np.sum(np.abs(gabor))

	return gabor


# Use Gabor filter to act on the image
def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    gray = np.pad(gray, (K_size//2, K_size//2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)
        
    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y : y + K_size, x : x + K_size] * gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


# Use 6 Gabor filters with different angles to perform feature extraction on the image
def Gabor_process(img):
    # get shape
    H, W, _ = img.shape

    # gray scale
    gray = BGR2GRAY(img).astype(np.float32)

    # define angle
    #As = [0, 45, 90, 135]
    As = [0,30,60,90,120,150]

    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = Gabor_filtering(gray, K_size=9, Sigma=1.5, Gamma=1.2, Lambda=1, angle=A)

        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out

def getImages(category,path):
    images=[]
    allImages= os.listdir(path)
    i=0
    for img2 in allImages:
            ##img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        img2 = cv2.imread(os.path.join(path,img2)).astype(np.float32)
        res = cv2.resize(img2, dsize=(32, 32))
        images.append([Gabor_process(res),category])
        i=i+1
        """
        if(i==120):
            return images
        """
    return images

def get_euclidean_distance2(data1,data2,lenght):
    distance=np.subtract(data1,data2)
    distance=np.square(distance)
    result=np.sum(distance)
    return (result**0.5)

#I do not use this function to reduce computational time. I use the above function.
def get_euclidean_distance(data1,data2,lenght):
    distance=0
    for i in range(lenght):
        distance=distance+ ((int(data1[i])-int(data2[i]))**2)
    return (distance**0.5)

    
def predict_weighted_result(target,index):
    strCovid="COVID"
    strNormal="NORMAL"
    strViral="Viral Pneumonia"
    nCovid=0
    nNormal=0
    nViral=0
    counter=0
    for i in index:        
        if(target[i]==strCovid):
            nCovid=nCovid+weighted_distance[counter]
        if(target[i]==strNormal):
            nNormal=nNormal+weighted_distance[counter]
        if(target[i]==strViral):
            nViral=nViral+weighted_distance[counter]
        counter=counter+1
    result=[]
    result.append(nCovid)
    result.append(nNormal)
    result.append(nViral)

    #if there is no majority; priority is viral ,normal covid
    if(max(result)==nViral):
        return strViral
    elif(max(result)==nNormal):
        return strNormal
    else:
        return strCovid
    
from heapq import nsmallest
#heapq to find lowest k elementh of list
#this function returns target indexes from given instance.
def get_neighbours(dataset,instance,k):
    distance_list=[]
    for i in dataset:
        distance_list.append(get_euclidean_distance2(i,instance,dataset.shape[1]))
    n_smallest_list=nsmallest(k, distance_list)
    index=[]
    for i in n_smallest_list:
        index.append(distance_list.index(i))
    return index      

def get_weighted_neighbours(dataset,instance,k,weighted_distance):
    distance_list=[]
    for i in dataset:
        result=get_euclidean_distance2(i,instance,dataset.shape[1])
        distance_list.append(result)
    n_smallest_list=nsmallest(k, distance_list)

    for i in n_smallest_list:
        if(i!=0):
            weighted_distance.append(1/i)
        else:
            weighted_distance.append(1)
    #print(n_smallest_list)
    index=[]
    for i in n_smallest_list:
        index.append(distance_list.index(i))
    return index   


def predict_result(target,index):
    strCovid="COVID"
    strNormal="NORMAL"
    strViral="Viral Pneumonia"
    nCovid=0
    nNormal=0
    nViral=0
    for i in index:
        if(target[i]==strCovid):
            nCovid=nCovid+1
        if(target[i]==strNormal):
            nNormal=nNormal+1
        if(target[i]==strViral):
            nViral=nViral+1
    result=[]
    result.append(nCovid)
    result.append(nNormal)
    result.append(nViral)
#if there is no majority; priority is viral ,normal covid
    if(max(result)==nViral):
        return strViral
    elif(max(result)==nNormal):
        return strNormal
    else:
        return strCovid
    
    
def get_predict(data,target,k):
    y_pred=[]
    for i in data:
        li=get_neighbours(x_train_data,i, k)
        predict = predict_result(target,li)

        y_pred.append(predict)
    return y_pred

def get_weighted_predict(data,target,k,weighted_distance):
    y_pred=[]
    
    for i in data: 
        li=get_weighted_neighbours(x_train_data,i, k,weighted_distance)
        predict = predict_weighted_result(target,li)
        y_pred.append(predict)
        del weighted_distance[:]
    return y_pred


def get_accuracy(cm):
    total = 0
    nTruePredict=0
    for i in range(3):
        for j in range(3):
            total=total+cm[i][j]
            if(i==j):
                nTruePredict=nTruePredict+cm[i][j]
    accuracy=100 * nTruePredict / total
    return accuracy


"""
Read train images. I ran it once and saved it with the pickle library 
in order not to start over and over each time the code was run.

DATADIR="C:/Users/gcand/.spyder-py3/train/train/"
CATAGORIES = ["COVID","NORMAL","Viral Pneumonia"]


covid_path = os.path.join(DATADIR,"COVID")
normal_path = os.path.join(DATADIR,"NORMAL")
pneumonia_path = os.path.join(DATADIR,"Viral Pneumonia")
#allImages= os.listdir(covid_path)

covid_images=[]
normal_images=[]
pneumonia_images=[]

covid_images=getImages("COVID",covid_path)
normal_images=getImages("NORMAL",normal_path)
pneumonia_images=getImages("Viral Pneumonia",pneumonia_path)

dataset=covid_images
dataset.extend(normal_images)
dataset.extend(pneumonia_images)
"""
import pickle
filename = 'dataset'
infile = open(filename,'rb')
dataset = pickle.load(infile)
infile.close()

target=[]
data=[]

# Shuffle dataset to cross validation
import random
random.shuffle(dataset)
accuracy_list=[]
benim_list=[]
k_fold=5
data_size=len(dataset)
test_size=int(data_size/k_fold)

j=0
weighted_distance=[]
import copy
from sklearn.metrics import confusion_matrix
average_accuracy=0
#for i in range(1,k_fold+1):
best_average=0
best_k=0   
#for i in range(1,k_fold+1,1):

"""
# Reading test data
# fixing the datapath for the code to work
DATADIR="C://Users//gcand//Desktop//bbm409-assignment1"
test_path = os.path.join(DATADIR,"test")
test_icin=[]
test_images=getImages("test",test_path)
for z in range(len(test_images)):
    test_icin.append(test_images[z][0])
    if(z==(len(test_images)-1)):
        test_icin = np.array(test_icin)
        test_icin=np.reshape(test_icin,(len(test_icin), 32*32))
"""           
test_target=pd.read_csv("submission-1.csv")
test_target=test_target.iloc[:,-1].values
test_target=test_target.astype(str)
test_target = np.array(test_target)

"""
filename = 'test_dataset'
outfile = open(filename,'wb')
pickle.dump(test_icin,outfile)
outfile.close()
"""

#Read test data from file
filename = 'test_dataset'
infile = open(filename,'rb')
test_icin = pickle.load(infile)
infile.close()


av_knn=0
av_knnList=[]
av_wknn=0
av_wknnList=[]
#K fold cross validation operatin. We chose K to be 5
for i in range(1,k_fold+1,1):  
    print("Step ",i," for k fold cross validation...")
    dataset_copy=copy.deepcopy(dataset)
    test_data=dataset_copy[((i-1)*test_size):test_size*i]
    del dataset_copy[((i-1)*test_size):test_size*i]
    train_data=dataset_copy
    x_test_data=[]
    y_test_data=[]
    for k in range(len(test_data)):
        x_test_data.append(test_data[k][0])
        y_test_data.append(test_data[k][1])
        if(k==(len(test_data)-1)):
            x_test_data = np.array(x_test_data)
            x_test_data=np.reshape(x_test_data,(len(test_data), 32*32))
            y_test_data = np.array(y_test_data)
    x_train_data=[]
    y_train_data=[]
    
    for j in range(len(train_data)):
        x_train_data.append(train_data[j][0])
        y_train_data.append(train_data[j][1])
        if(j==(len(train_data)-1)):
            x_train_data = np.array(x_train_data)
            x_train_data=np.reshape(x_train_data,(len(train_data), 32*32))
            y_train_data = np.array(y_train_data)
            
    """       
    #predict from validation data       
    y_pred=get_predict(x_test_data, y_train_data,3)
    print(confusion_matrix(y_test_data,y_pred))
    print("Accuracy for k=",3," => ",get_accuracy(confusion_matrix(y_test_data,y_pred)))
    av_knnList.append(get_accuracy(confusion_matrix(y_test_data,y_pred)))
    
    y_pred2=get_weighted_predict (x_test_data, y_train_data,3,weighted_distance)
    print(confusion_matrix(y_test_data,y_pred2))
    print("Accuracy for k=",3," => ",get_accuracy(confusion_matrix(y_test_data,y_pred2)))
    av_wknnList.append(get_accuracy(confusion_matrix(y_test_data,y_pred2)))
    """ 
           
    #Predict from test data...
    #we specified that best k value is 3 . So I used k=3 for knn and weighted knn
    y_pred=get_predict(test_icin, y_train_data,3)
    cm=confusion_matrix(test_target,y_pred)
    cm=np.array(cm)

    #The confusion matrix comes as a false 4x4 matrix. The 3th row and 3th row always come as 0.
    #This row and column is deleted. When the row and column filled with 0 are deleted, the correct cm is found.
    cm=np.delete(cm,3,0)
    cm=np.delete(cm,2,1)
    
    print(cm)
    print("Accuracy for KNN : ", get_accuracy(cm))
    av_knn=get_accuracy(cm) + av_knn
    av_knnList.append(get_accuracy(cm))
    
    y_pred2=get_weighted_predict(test_icin,y_train_data,3,weighted_distance)

    cm2=confusion_matrix(test_target,y_pred2)
    cm2=np.array(cm2)
    cm2=np.delete(cm2,3,0)
    cm2=np.delete(cm2,2,1)      
         
    print(cm2)
    print("Accuracy for Weighted-KNN : ",get_accuracy(cm2))   
    av_wknn=av_wknn+get_accuracy(cm2)
    av_wknnList.append(get_accuracy(cm2))
    
 
     
print("------------------summary---------------------------")
print("Average KNN Accuracy : ",av_knn/5)          
print("Average Weighted-KNN Accuracy : ",av_wknn/5)            
        

"""
    Kaggle
    #To write predict to csv file.
    counter=0
    import csv
    with open('sonuc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Id","Category"])
        for pr in y_pred2:
            if(pr=="Viral Pneumonia"):
                writer.writerow([counter+1, "VIRAL"])
            else:
                writer.writerow([counter+1, y_pred2[counter]])
            counter=counter+1
"""  

        
"""  
    #To determine the k value that gives the highest accuracy 

    for t in range(1,20,2):        
        y_pred=get_predict(x_test_data, y_train_data, t)
        print(confusion_matrix(y_test_data,y_pred))
        print("Accuracy for k=",t," => ",get_accuracy(confusion_matrix(y_test_data,y_pred)))
        if(get_accuracy(confusion_matrix(y_test_data,y_pred))>best_average):
            best_average=get_accuracy(confusion_matrix(y_test_data,y_pred))
            best_k=t
        average_accuracy=average_accuracy+get_accuracy(confusion_matrix(y_test_data,y_pred))
       
    print("------------------summary---------------------------")
    #average_accuracy=average_accuracy+get_accuracy(confusion_matrix(y_test_data,y_pred))
    #average_accuracy=average_accuracy/k_fold
    average_accuracy=average_accuracy/10
    print("Average Accuracy : ",average_accuracy)
    print("Best Accuracy : {} with k value = {}".format(best_average, best_k))
    accuracy_list.append(average_accuracy)
    benim_list.append(best_k)
    average_accuracy=0
    best_average=0
    best_k=0  
    """
       
end = time.time()
print("Gecen zaman",(end-start)/60," dakika")
