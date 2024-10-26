#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from os import listdir
from os.path import isfile, join


# In[5]:


benign_dir = "E:\\skin\\train\\benign"       
malignant_dir = "E:\\skin\\train\\malignant"                                                            


# In[6]:


benign_files = [f for f in listdir(benign_dir) if isfile(join(benign_dir, f))]
malignant_files = [f for f in listdir(malignant_dir) if isfile(join(malignant_dir, f))]


# In[7]:


X = []
y = []


# In[8]:


for file in benign_files:
    img = plt.imread(join(benign_dir, file))
    X.append(img)
    y.append(0)  # 0 for Benign

for file in malignant_files:
    img = plt.imread(join(malignant_dir, file))
    X.append(img)
    y.append(1)  # 1 for Malignant

X = np.array(X)
y = np.array(y)


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
 


# In[10]:


# In[ ]:
from skimage.transform import resize

X_train_resized = []
for img in X_train:
    X_train_resized.append(resize(img, (64, 64, 3)))

X_test_resized = []
for img in X_test:
    X_test_resized.append(resize(img, (64, 64, 3)))

X_train_resized = np.array(X_train_resized)
X_test_resized = np.array(X_test_resized)


# In[10]:


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_test_resized),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_test_resized[i])    
    plt.axis('off')


# In[11]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_history = model.fit(X_train_resized, y_train, epochs=10, batch_size=64, validation_data=(X_test_resized, y_test))


# In[12]:



# SVM Model
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
svm_predict = svm_model.predict(X_test.reshape(X_test.shape[0], -1))
svm_accuracy = accuracy_score(y_test, svm_predict)


# In[57]:


import pickle
filename = 'CNN_model.sav'
pickle.dump(model, open(filename, 'wb'))


# # FOR TESTING

# In[1]:


#LOAD MODEL ON RUNNING
import pickle

filename = 'CNN_model.sav'
model = pickle.load(open(filename, 'rb'))

filename2 = 'svm_model.sav'
svm_model = pickle.load(open(filename2, 'rb'))


# In[25]:


# Evaluation
print('CNN Accuracy:', model.evaluate(X_test_resized, y_test))
print('SVM Accuracy:', svm_accuracy)


# In[74]:


cnn_predict = (model.predict(X_test_resized) > 0.5).astype("int32")
cnn_cm = confusion_matrix(y_test, cnn_predict)
svm_cm = confusion_matrix(y_test, svm_predict)


# In[30]:


print("CNN Confusiopn Matrix:\n", cnn_cm)
print("SVM Confusion Matrix:\n", svm_cm)


# In[34]:


from sklearn.metrics import confusion_matrix, classification_report

y_pred_cnn = model.predict(X_test_resized)
y_pred_cnn_binary = (y_pred_cnn > 0.5).astype(int)  

cm_cnn = confusion_matrix(y_test, y_pred_cnn_binary)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix - CNN Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Classification Report - CNN Model")
print(classification_report(y_test, y_pred_cnn_binary))


# In[28]:


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('CNN Training/Validation Accuracy')
plt.plot(cnn_history.history['accuracy'], label='Training')
plt.plot(cnn_history.history['val_accuracy'], label='Validation')
plt.legend()

plt.subplot(1, 2, 2)givce
plt.title('CNN Training/Validation Loss')
plt.plot(cnn_history.history['loss'], label='Training')
plt.plot(cnn_history.history['val_loss'], label='Validation')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Assuming you have loaded and preprocessed your data into X_train, y_train, X_val, y_val

# Flatten the training and validation data
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_test.reshape(X_test.shape[0], -1)

svm_val_predict = svm_model.predict(X_val_flat)
svm_val_accuracy = accuracy_score(y_val, svm_val_predict)
print("SVM Validation Accuracy:", svm_val_accuracy)

# Plotting validation accuracy and loss
plt.figure(figsize=(10, 5))

# Plotting validation accuracy
plt.subplot(1, 2, 1)
plt.title('SVM Validation Accuracy')
plt.plot(svm_accuracy, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plotting validation loss
plt.subplot(1, 2, 2)
plt.title('SVM Validation Loss')
plt.plot(svm_loss, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[68]:


from skimage.transform import resize
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

new_image_path = "test1.jpeg"
new_image = Image.open(new_image_path)
new_image_resized = new_image.resize((64, 64))  
new_image_array = np.array(new_image_resized) 
new_image_array = new_image_array / 255.0 

new_image_array = np.expand_dims(new_image_array, axis=0)
prediction = model.predict(new_image_array)

plt.imshow(new_image_resized)
plt.axis('off')

if prediction[0][0] > 0.5:
    plt.title("CNN MODEL : Prediction: Class 1 : Melignant")
else:
    
    plt.title("CNN MODEL : Prediction: Class 0 : Benign")

plt.show()


# In[69]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

new_image_path = "melanoma_7.jpg"
new_image = Image.open(new_image_path)

new_image_array = np.array(new_image) 
new_image_array = new_image_array / 255.0 

new_image_array_flattened = new_image_array.reshape(1, -1)

svm_predict = svm_model.predict(new_image_array_flattened)

plt.imshow(new_image)
plt.axis('off')

if svm_predict[0] == 1:
    plt.title("SVM MODEL : Prediction: Class 1 : Malignant")
else:
    plt.title("SVM MODEL : Prediction: Class 0 : Benign")

plt.show()

