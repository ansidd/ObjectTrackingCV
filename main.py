import cv2

cap = cv2.VideoCapture("./video2.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=30)

tracker_history = {}

obj_count=0

def e_dist(a,b):
    dist = 0
    for i in range(len(a)):
        dist += (a[i]-b[i])**2
    dist = dist**0.5
    return dist
        
        
def track_objs(curr_positions, history):
    global obj_count
    
    if len(history)==0:
        n_objects = len(curr_positions)
        obj_count = n_objects
        history = list(zip(range(n_objects), curr_positions))
        return list(range(n_objects)), history
    else:
        object_ids = []
        for obj_pos in curr_positions:
            xi,yi,wi,hi = obj_pos
            for i,obj in enumerate(history):
                xj,yj,wj,hj = obj[1]
                if e_dist((xi,yi,wi,hi), (xj,yj,wj,hj))<100 and obj[0]!=-1:
                    object_ids.append(obj[0])
                    break
                    
                if (i+1)==len(history):
                    obj_count+=1
                    object_ids.append(obj_count)
        history = list(zip(object_ids, curr_positions))
        return object_ids, history

while True:
    ret, frame = cap.read()
    
    
    if ret:
        mask = object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cnt_positions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 600:
                #cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                cnt_positions.append([x, y, w, h])
                
        object_ids, tracker_history = track_objs(cnt_positions, tracker_history)
        for i in range(len(cnt_positions)):
            if object_ids[i]==-1:
                continue
            x,y,w,h = cnt_positions[i]
            cv2.putText(frame,  str(object_ids[i]), (x, y-15),cv2.FONT_HERSHEY_PLAIN,1, (255,0,0))
            cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (0,255,0), 2)
                
        
        cv2.imshow("Frame", frame)
        #cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)

    if key==27:
        break
        

            
                    

cap.release()
cv2.destroyAllWindows()

    