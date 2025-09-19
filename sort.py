from kalmanFilter import KalmanFilter
import manual_written_hit_detector
import cv2
import statistics
import math
import numpy as np


#Code for my SORT algorithm implementation to keep track of what fighters are each specific fighter, infers the path of each fighter, where they are going to be next


#This code is dependent on the manual_written_hit_detector.py code, as it will be used in conjunction with it to keep track of the fighters


class sort(manual_written_hit_detector.copy):
    fighterOneTicker = 0
    fighterTwoTicker = 0

    self.trackers = {
        0: [],  #fighter one trackers
        1: []  #fighter two trackers
    }


    def __init_z_(self):
        super().__init__()
        # Initialize SORT specific parameters here
        # e.g., self.trackers = []
        # e.g., self.next_tracker_id = 0z
        print("SORT algorithm initialized.")   

        #Initialize Kalman Filters for each fighter's keypoints
        num_keypoints = 17
        for fighter_id in [0, 1]:
            self.trackers[fighter_id] = [
                KalmanFilter(dt=1.0, u_x=0, u_y=0, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
                for _ in range(num_keypoints)
            ]

        self.updateKalman()





        
        
    def hungarianFunction(fighter0Keypoints, fighter1Keypoints):
        print("This is a placeholder for the Hungarian function.")
        # Implement Hungarian algorithm logic here
        #This function will assign detected bounding boxes to existing trackers based on the minimum cost (distance)
        #This function will help in associating detections to the correct fighters over time
        #This function will help in maintaining the identity of each fighter across frames
        #This function will help in reducing identity switches and improving tracking accuracy
        #Returns which fighter 0 or 1

        predictionFighter0, predictionFighter1 = predictionFromKalman()

        costMatrix = [[0, 0], [0, 0]]  #2D list for cost matrix


        # Fighter 0 detection vs Fighter 0 prediction
        costMatrix[0][0] = math.sqrt(
            (fighter0Keypoints.x - predictionFighter0[0][0]) ** 2 +
            (fighter0Keypoints.y - predictionFighter0[1][0]) ** 2
        )

        # Fighter 0 detection vs Fighter 1 prediction
        costMatrix[0][1] = math.sqrt(
            (fighter0Keypoints.x - predictionFighter1[0][0]) ** 2 +
            (fighter0Keypoints.y - predictionFighter1[1][0]) ** 2
        )

        # Fighter 1 detection vs Fighter 0 prediction
        costMatrix[1][0] = math.sqrt(
            (fighter1Keypoints.x - predictionFighter0[0][0]) ** 2 +
            (fighter1Keypoints.y - predictionFighter0[1][0]) ** 2
        )

        # Fighter 1 detection vs Fighter 1 prediction
        costMatrix[1][1] = math.sqrt(
            (fighter1Keypoints.x - predictionFighter1[0][0]) ** 2 +
            (fighter1Keypoints.y - predictionFighter1[1][0]) ** 2
        )


        # Use Hungarian algorithm to find the optimal assignment

        smallest = int.max

        for i in range(len(costMatrix)):
            for j in range(len(costMatrix[i])):
                if(costMatrix[i][j] < smallest):
                    smallest = costMatrix[i][j]


        if(smallest == costMatrix[0][0] or smallest == costMatrix[1][1]):
            return 0 # correct assignment
        else:
            return 1 # swapped assignment







    def trackStatistics():






    def processVideo(self):
        results = self.model.predict(source=validated_path, conf=0.7, save=False, save_txt=False, show = True)

        self.trackStatisitics(results)



    def R_from_conf(conf, R_base=5.0):
        """
        Scale measurement noise covariance by confidence.
        Lower conf -> larger noise (less trust).
        """
        conf = np.clip(conf, 0.01, 1.0)
        return np.eye(2) * (R_base / conf)
    

    def predictionFromKalman():
        predictionsFighter0 = []
        predictionsFighter1 = []
        for kf in self.trackers[0]:
            pred = kf.predict()
            predictionsFighter0.append(pred)

        for kf in self.trackers[1]:
            pred = kf.predict()
            predictionsFighter1.append(pred)

        return predictionsFighter0, predictionsFighter1        


    def updateKalman():




    







def main():
    print("Starting MMA Hit Detector...")
    video_path = input("Enter in a video path: ")
    mmaProcessingInstance = mmaProcessing(video_path)
    results = mmaProcessingInstance.processVideo()
    





if __name__ == "__main__":
    main()

