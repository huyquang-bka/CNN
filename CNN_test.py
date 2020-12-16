import numpy as np
import cv2
import pickle
import os
from sklearn.metrics import accuracy_score
from shutil import copyfile
from keras.models import model_from_json
from skimage import transform

#### LOAD THE TRAINNED MODEL

pickle_in = open("A/model_size32_filter64_agps.p", "rb")
model = pickle.load(pickle_in)


#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


accQuang = []
accOld = []
folderPath = ['full3/Test_full3/C9_agps/free/3', 'full3/Test_full3/C9_agps/busy/3']
for path in folderPath:
    imagePath = os.listdir(path)
    n = len(imagePath)
    predictLabel = []
    count = 0
    check = 0
    if 'free' in path:
        check = 0
    else:
        check = 1

    for image in imagePath:
        img = cv2.imread(path + '/' + image)
        img = cv2.resize(img, (32, 32))
        img = preProcessing(img)
        img = img.reshape(1, 32, 32, 1)
        classIndex = int(model.predict_classes(img))
        # predictions = model.predict(img)
        # probVal = np.amax(predictions)
        predictLabel.append(classIndex)
        if classIndex == 0 and check == 1:
            copyfile(f"{path}/{image}", f"Wrong/busy2free/{image}")
        elif classIndex == 1 and check == 0:
            copyfile(f"{path}/{image}", f"Wrong/free2busy/{image}")
        count += 1
        print(f'Loading: {count}/{n}')

    realLabel_Quang = []
    if check == 0:
        realLabel_Quang = np.zeros_like(predictLabel)
    else:
        realLabel_Quang = np.ones_like(predictLabel)
    accuracy_Quang = accuracy_score(predictLabel, realLabel_Quang)

    ################   Old weight   ######################

    json_file = open('FileOld/modelC9f.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    Loaded_Model = model_from_json(loaded_model_json)
    # Load weights into new model
    Loaded_Model.load_weights("weightC9_agps.h5")
    print("Loaded")

    Predict_Slot = []
    filename_array = []
    count = 0
    for filename in os.listdir(path):
        images = []
        img = cv2.imread(os.path.join(path, filename))
        images.append(transform.resize(img, (150, 150, 3)))
        images = np.array(images)
        if Loaded_Model.predict(images)[0][0] > 0.5:
            Predict_Slot.append(1)
            if check == 0:
                copyfile(f"{path}/{filename}", f"Wrong/free2busy_old/{filename}")
        else:
            Predict_Slot.append(0)
            if check == 1:
                copyfile(f"{path}/{filename}", f"Wrong/busy2free_old/{filename}")

        count += 1
        print(f"Loading: {count}/{len(os.listdir(path))}")

    realLabel_old = []
    if check == 1:
        realLabel_old = np.ones_like(Predict_Slot)
    else:
        realLabel_old = np.zeros_like(Predict_Slot)
    accuracy_old = accuracy_score(Predict_Slot, realLabel_old)

    accQuang.append(accuracy_Quang)
    accOld.append(accuracy_old)

print("Acc Quang: ", accQuang)
print("Acc Old: ", accOld)
