import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
import cv2
def webcam():
    
    camera = cv2.VideoCapture(0)
    count=0
    while True:
        return_value,image = camera.read()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cv2.imshow('image',gray)

        ## Take photo
        if cv2.waitKey(1)& 0xFF == ord('s'):

            ret,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
            #thresh = thresh[312:712, 230:630]
            thresh = cv2.resize(thresh, (50, 50))
            
            
            ## Save Frame
            name =r'C:\Users\Neha Manivannan\AppData\Local\Programs\Python\Python36\img' + str(count) + '.png'
            print ('Creating:' + name)
            cv2.imwrite(name, thresh)
 

            count+=1
            continue

        ## Close
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break

        
    camera.release()
    cv2.destroyAllWindows()

def merge_pictures():
    images_list = []
    for i in range(2):
       img = ("img%d.png" %(i))
       images_list.append(img)
    imgs = [ Image.open(i) for i in images_list ]
    print(len(imgs))
#Find the smallest image, and resize the other images to match it
    listed =[0,1]
    i=0
    j=0
    while i<2:
       if len(listed)>2:
         listed = [0,1]
       listed[0] = imgs[i]
       listed[1]= imgs[i+1]
       min_img_shape = sorted( [(np.sum(k.size), k.size ) for k in listed])[0][1]
       img_merge = np.hstack( (np.asarray( p.resize(min_img_shape,Image.ANTIALIAS) ) for p in listed ) )
       i = i+2
       img_merge1 = Image.fromarray( img_merge)
       res = ('img_%d.png' %j)
       img_merge1.save(res)
       j=j+1
# save the horizontally merged images

def csvfile():
    for i in range(1):
        file = ("img_%d.png" %(i))
        img= Image.open(file)
        pixels = np.asarray(img)
        pixels = pixels/255
        csvfile=("testdata%d.csv" %(i))
        np.savetxt(csvfile ,pixels.reshape(1,pixels.size),delimiter=",")
    list = []
    for j in range(1):
        name =  ('testdata%d.csv' %(j))
        list.append(name)
    combined_csv = pd.concat( [ pd.read_csv(f, header = None) for f in list] )    
    combined_csv.to_csv( "testor.csv", index=False ,header=None)

def classifier():
    feature = pd.read_csv('training3.csv',header = None)
    feature= feature.values
    label = pd.read_csv('traininglabelnew.csv',header = None)
    label=label.values
    testfeature = pd.read_csv('testor.csv',header = None)
    testfeature=testfeature.values
    testlabel = pd.read_csv('testlabelsnew.csv',header = None)
    testlabel=testlabel.values
    training_digits_pl = tf.placeholder("float", [None, 5000])
    test_digit_pl = tf.placeholder("float", [5000])
    l1_distance = tf.abs(tf.add(training_digits_pl, tf.negative(test_digit_pl)))
    distance = tf.reduce_sum(l1_distance, axis=1)
    pred = tf.argmin(distance, 0)
    accuracy = 0.
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
         sess.run(init)
# loop over test data
         for i in range(len(testfeature)):
# Get nearest neighbor
            nn_index = sess.run(pred, feed_dict={training_digits_pl: feature, test_digit_pl: testfeature[i, :]})

# Get nearest neighbor class label and compare it to its true label
            print("Test", i, "Prediction:", np.argmax(label[nn_index]),  "True Label:",  np.argmax(testlabel[i]))
# Calculate accuracy
            if np.argmax(label[nn_index]) == np.argmax(testlabel[i]):
                  accuracy += 1./len(testfeature)

         print("Done!")
         print("Accuracy:", accuracy)
webcam()
merge_pictures()
csvfile()
classifier()



    
