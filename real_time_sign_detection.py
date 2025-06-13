import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard 
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model


mp_holistic=mp.solutions.holistic           # Holistic model
mp_drawing=mp.solutions.drawing_utils       # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #color conversions BGR 2 RGB
    image.flags.writeable = False                   #image no longer writable to save memory
    results = model.process(image)                  # make prediction
    image.flags.writeable = True                    # make image writable
    image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   #color conversions back RGB 2 BGR
    return image,results

def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp.solutions.face_mesh.FACEMESH_TESSELATION)         # Draw face conncetions
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)                        # Draw pose conncetions
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)                   # Draw left hand connections
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)                  # Draw right hand connections


def draw_styled_landmarks(image,results):                                                                       # Just for formatting the landmarks

    # Draw face conncetions
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp.solutions.face_mesh.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1))     

    # Draw pose conncetions
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)    

    # Draw left hand connections
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)

    # Draw right hand connections   
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)  


def extract_keypoints(results):
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)       # Cause each hand has 21 landmarks with 3 values of x, y and z
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)              # Cause face have 468 landmarks and 3 values x y z
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([pose,face,lh,rh])



cap = cv2.VideoCapture(0)

# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:


    
    #path for exported data, numpy arrays
    DATA_PATH = os.path.join('MP_DATA')

    # Actions that we try to detect
    actions = np.array(['hello','thanks','iloveyou','Sorry','How are you'])

    #Thirty videos worth of data
    no_sequence = 30

    # Videos are going to be 30 frames in length
    sequence_length = 30

    label_map = {label:num for num, label in enumerate(actions)}

    ######################################################################################

    # To collect data

    '''
    for action in actions:
        for sequence in range(no_sequence):
            try:
                os.makedirs(os.path.join(DATA_PATH, action,str(sequence)))
            except:
                pass

    
    
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequence):
            #Loop through video length aka sequence length
            for frame_num in range(sequence_length):

        

                # Read feed
                ret, frame = cap.read()

                #Make detections
                image, results= mediapipe_detection(frame,holistic)
                
                
                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Apply collection logic
                if frame_num == 0:                                                                                        # If we are at frame 0
                    cv2.putText(image, 'SHARING COLLECTION', (120,200),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)                                       
                    cv2.putText(image, 'Collecting frames for {} Video Numeber {}'.format(action, sequence),(15,12),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                    cv2.waitKey(2000)                                                                                     # Wait 2 seconds
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Numeber {}'.format(action, sequence),(15,12),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)       

                # Show to screen
                cv2.imshow('OpenCV Feed', image)


            

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action,str(sequence), str(frame_num))
                np.save(npy_path, keypoints)      

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):  # If 'q' is pressed
                    break

    
    # Move these OUTSIDE the loop
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

    '''

    ##########################################################################

    sequences, labels =[],[] # Two blank arrays
    for action in actions:    
        for sequence in range(no_sequence):  
            window=[]                                                                                       # going to represent all of the frames for a particular video
            for frame_num in range(sequence_length):                                                        # Looping through each frame
                res=np.load(os.path.join(DATA_PATH, action,str(sequence), "{}.npy".format(frame_num)))      # Loading that frame
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    
    X = np.array(sequences)           
    Y = to_categorical(labels).astype(int)                                                                  # Basically jaise pdha tha ki array mein hello ko first position [1,0,0] then thanks ko second [0,1,0] and ilu ko third [0,0.1]

    X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.05)                                   # Test partition is going to be 5% of our data

    
    ###################################################################

    #Building and Training LSTM Neural Network
    
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0],activation='softmax'))

    # For compiling the model
    
    #model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    #model.fit(X_train,Y_train,epochs=500, callbacks=[tb_callback])
    
    ####################################################################



    # Making Predictions

    test_preds= model.predict(X_test)

    # Saving Model

    #model.save('action.h5')            # If already saved once use load_model              

    model = load_model('action.h5') 
    yhat=model.predict(X_test)
    ytrue = np.argmax(Y_test, axis=1).tolist()
    yhat = np.argmax(yhat,axis=1).tolist()

    #####################################################################

    # Testing 

    colors = [(246,272,12),(262,232,12),(322,472,113),(573,958,12),(423,88,298)]
    def prob_viz(res,actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40),(int(prob*100),90+num*40), colors[num], -1)
            cv2.putText(output_frame,actions[num],(0,85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2,cv2.LINE_AA)
        
        return output_frame

    # Detection
    sequence = []
    sentence = []
    threshold = 0.95
    res = np.array([0,0,0])

    while  cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        #Make detections
        image, results= mediapipe_detection(frame,holistic)
                
                
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints=extract_keypoints(results)
        sequence.insert(0,keypoints)
        sequence = sequence[:30]

        if len(sequence) == 30:
            input_data = np.expand_dims(sequence, axis=0)  # shape (1, 30, 1662)
            res = model.predict(input_data)[0]             # shape (3,)
            print("Prediction:", res)
            print("Detected Action:", actions[np.argmax(res)])


        # .3 Viz logic
        
        if res[np.argmax(res)] > threshold:
            if len(sentence) > 0:
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5:
            sentence = sentence[-5:]


        # Viz Probabilites
        image=prob_viz(res,actions,image,colors)

        cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)


        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):  # If 'q' is pressed
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows