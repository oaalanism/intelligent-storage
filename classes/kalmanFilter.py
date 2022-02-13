import numpy as np

class KalmanFilter:
     """
     This object implements kalman filter
     
     For more information consult next link: https://thinkautonomous.medium.com/computer-vision-for-tracking-8220759eee85
    
     Parameters
     ----------
        x : Int
            x center coordinate of bounding box
        y : Int
            y center coordinate of bounding box
        w : Int
            width of bounding box
        h : Int
            height of bounding box
     """
    def __init__(self, x=None, y=None, w=None, h=None):
        if x != None and y != None and h != None and w != None:
            self.init(x, y, w, h)
            
    def update(self, Z):
        """
         Update function of kalman flter

         Parameters
         ----------
            z : Int
                measure of bounding box
         Returns
        ----------
        """

        Y = Z - np.matmul(self.H, self.X_)
        S = np.matmul(self.H, np.matmul(self.P_, self.H_T))  + self.R
        K = np.matmul(self.P_, np.matmul(self.H_T, np.linalg.inv(S)))
        self.X = self.X_ + np.matmul(K, Y)
        
        self.P = np.matmul((self.I - np.matmul(K, self.H)), self.P_)

    def predict(self):
        """
         Prediction function state of Kalman Filter

         Parameters
         ----------
         Returns
         ----------
        """
        
        self.X_ = np.matmul(self.F, self.X)  
        self.P_ = np.matmul(np.matmul(self.F, self.P), np.transpose(self.F)) + self.Q

    def init(self, x, y, w, h):
        """
         Function to initialize Kalman Filter

         Parameters
         ----------
            x : Int
            x center coordinate of bounding box
            
            y : Int
                y center coordinate of bounding box
                
            w : Int
                width of bounding box
                
            h : Int
                height of bounding box
         Returns
         ----------
            X: Array
                Initial state
            F: Array
                State transition
            P: Array
                Predicted matrix
            Q: Array
                Process Covariance Matrix
        """
        self.X = np.array([x, y, w, h, 0, 0, 0, 0])

        #self.F = np.eye(8) + np.eye(8, k=4)*0.03

        dt = 0.03

        self.F = np.array([
                        [1, 0, 0, 0, dt, 0, 0, 0],
                        [0, 1, 0, 0, 0, dt, 0, 0],
                        [0, 0, 1, 0, 0, 0, dt, 0],
                        [0, 0, 0, 1, 0, 0, 0, dt],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1]
                        ])
        
        self.Q = np.eye(8)*0.1

        self.P = np.eye(8)*10

        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                        ])

        self.H_T = np.transpose(self.H)

        self.R = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        self.I = np.eye(8)

        return self.X, self.F, self.P, self.Q


