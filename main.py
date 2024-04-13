import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Labels for determine movement servo
include_labels = ["backpack","umbrella","handbag","bottle","cup","fork","knife","spoon","banana","scissors"
         "carrot","hotdog","laptop","mouse","remote","keyboard","cellphone","book","clock","vase",
         "hairdrier","toothbrush"]
# Labels based on train
labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
         "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
         "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
         "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
         "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
         "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
         "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
         "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
         "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
         "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

# Load Model
model = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
# Get The Layer 
layers = model.getLayerNames()

# Stream the data
while True :
     # Read Retangle and frame of second data
     ret, frame = cap.read()
     # Determine the frame width and height
     frame_width = frame.shape[1]
     frame_height = frame.shape[0]
     # Preprocess the input frame
        # - Scales the pixel values
        # - resize the square size 416x416 pixels, swap the red & blue channels data
        # - Dont crop the realtime data frame 
     frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True, crop=False)
     # Config the pixel representation data
     colors = ["0,255,255","0,0,255","255,0,0","255,255,0","0,255,0"]
     colors = [np.array(color.split(",")).astype("int") for color in colors]
     colors = np.array(colors) 
     colors = np.tile(colors, (18,1)) 
     # Got the Output Layer from the Model
     output_layer = [layers[layer-1] for layer in model.getUnconnectedOutLayers()]
     # Set The input based on result from frame_blob
     model.setInput(frame_blob)
     # Forward Pass Model for classification process based on realtime process
     detection_layers = model.forward(output_layer)
     
     # Detection labels determination
     ids_list = []
     boxes_list = []
     confidences_list = []
     # Get the each layer on detection layers -> detection later <- object detection (useable)
     for detection_layer in detection_layers:
        for object_detection in detection_layer:
            # Get the score based on from 5 layer on the top
            scores = object_detection[5:] 
            # Predicted Id based on scores result
            predicted_id = np.argmax(scores) 
            confidence = scores[predicted_id] 
            
            # Detection object based on box
            if confidence > 0.35: 
                
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_width,frame_height,frame_width,frame_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                
                start_x = int (box_center_x - (box_width/2))
                start_y = int (box_center_y - (box_height/2))
     
                ids_list.append(predicted_id)
                confidences_list.append(float (confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
     
     max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
    # Detection object based on max_id
     for max_id in max_ids: 
            max_class_id = max_id 
            box = boxes_list[max_class_id] 
                    
            start_x = box[0] 
            start_y = box[1]
            box_width = box[2]
            box_height = box[3] 

            predicted_id = ids_list[max_class_id] 
            
            label =  labels[predicted_id] 
            check = label in include_labels
            if check is True:
                confidence = confidences_list[max_class_id]
                end_x = start_x + box_width
                end_y = start_y + box_height
                    
                box_color = colors[predicted_id]
                box_color = [int(each) for each in box_color]
                
                label = "{}: {:.2f}%".format(label, confidence * 100)
                print ("predicted object {}".format(label))
                    
                    
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
                cv2.putText(frame, label, (start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color,2)
                label =  labels[predicted_id] 
     cv2.imshow("Real Time Object Detection v2", frame)

     if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()