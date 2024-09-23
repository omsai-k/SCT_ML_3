import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

train_data = "/Users/omsai/Documents/OS/SkillCraft/DATA/train"
test_data = "/Users/omsai/Documents/OS/SkillCraft/DATA/test1"

def load_train_images(folder):
    image_data = []
    image_labels = []
    
    for filename in os.listdir(folder):
        if filename.startswith('cat'):
            label = 0
        elif filename.startswith('dog'):
            label = 1
        else:
            continue
    
        img_path = os.path.join(folder,filename)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img,(64,64))
        
        image_data.append(img_resized.flatten())
        image_labels.append(label)
        
    return np.array(image_data),np.array(image_labels)

def load_test_images(folder):
    image_data = []
    image_filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (64, 64))
        
        image_data.append(img_resized.flatten())
        image_filenames.append(filename)
    
    return np.array(image_data), image_filenames

Xtrain , Ytrain = load_train_images(train_data)

Xtest , Ytest = load_test_images(test_data)

model = svm.SVC(kernel='linear')
model.fit(Xtrain,Ytrain)

Ypred = model.predict(Xtest)

for filename, prediction in zip(Ytest,Ypred):
    label = 'dog' if prediction == 1 else 'cat'
    print(f"{filename}-->{label}")
