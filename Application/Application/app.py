from flask import Flask, render_template, jsonify, request,  redirect, Response
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns; sns.set()
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import random, json
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import os,base64
import requests as req
from io import BytesIO
from PIL import Image


app = Flask(__name__)

@app.route("/")
def main():
    return render_template('dashboard.html')

@app.route("/results", methods=['POST'])
def results ():
    data = request.get_json()
    result=''
    for item in data:
        name= item['name']
        gs=item['gs']
        vorp=item['vorp']        
        result += 'His name is '+str(name)+' ' + str(gs)+ ' '+str(vorp)+ '\n'
    return result


@app.route('/regression/results',methods=['POST'])
def regressor():
#    input = np.array([55,1727,11.3,1.7,1.8,0.5,1.2,204,532,115,323,89,209,45,60,170,192,59,83,568])

    
    
    data_all = pd.read_csv("Salary_Prediction/code_test/data_pg_v2.0.csv")
    X_all = data_all.iloc[:,4:]
    Y_salary = data_all['salary']

    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
    X_data = min_max_scaler.fit_transform(X_all)
    Y_data = min_max_scaler.fit_transform(Y_salary[:,np.newaxis])

    forest = RandomForestRegressor(n_estimators=1000, random_state=1, n_jobs=-1, min_samples_leaf = 10)
    forest.fit(X_data,Y_data.flatten())  
    data =request.get_json()
    for item in data:
        name= item['name']
        gs=item['gs']
        per=item['per']
        mp=item['mp']
        ows=item['ows']
        dws=item['dws']
        obpm=item['obpm']
        vorp=item['vorp']
        fg= item['fg']
        p3= item['p3']
        p3a= item['p3a']
        p2= item['p2']
        p2a = item['p2a']
        ft= item['ft']
        fta= item['fta']
        drb= item['drb']
        ast= item['ast']
        stl= item['stl']
        tov= item['tov']
        pts= item['pts']
        
#    X = min_max_scaler.fit_transform(input.reshape(-1, 1))
#    X_input = X.reshape(1,20)
#    Y_predict = forest.predict(X_input)  
#    Y_max = np.max(Y_salary)
#    Y_min = np.min(Y_salary)
#    Y = Y_predict * (Y_max-Y_min) + Y_min
#    Y= math.floor(Y)
        
        
    return 'The predicted salary value is $'


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
   
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


#def image_to_base64(image_path):
 #   img = Image.open(image_path)
  #  output_buffer = BytesIO()
   # img.save(output_buffer, format='JPEG')
    #byte_data = output_buffer.getvalue()
    #base64_str = base64.b64encode(byte_data)
    #return base64_str



def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
   
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    #pil_image.show()
   # response = req.get(pil_image）
    #image = Image.open(BytesIO(response.content)) 
    #ls_f=base64.b64encode(BytesIO(response.content).read())     
     #ls_f      #the output of the function
    return pil_image
 


@app.route('/img_result',methods=['POST'])
def facerec(img_path):
#if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    #print("Training KNN classifier...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    #print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    #for image_file in os.listdir("knn_examples/test"):
     #   full_file_path = os.path.join("knn_examples/test", image_file)

      #  print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
    predictions = predict(img_path, model_path="trained_knn_model.clf")

        # Print results on the console
    for name, (top, right, bottom, left) in predictions:
  #      print("- Found {} at ({}, {})".format(name, left, top))
          output_img=show_prediction_labels_on_image(img_path, predictions)
          output_img.save('outputimg.jpg')
          return output_img


          
         # base64_data = base64.b64encode(output_img)
        #  base64_data=image_to_base64(output_img)
       #   print(base64_data)
#          output_img.show()
     #    response = req.get(output_img）
      #  image = Image.open(BytesIO(response.content))
       # ls_f=base64.b64encode(BytesIO(response.content).read()) 
        #print(ls_f) 
        # Display results overlaid on an image
      #  show_prediction_labels_on_image(img_path, predictions)
      
    
#API test  (output:picture_base64)  
#if __name__ == "__main__":
#facerec("/Users/mac/Desktop/nba_test.jpeg")




if __name__ == "__main__":
    app.run()