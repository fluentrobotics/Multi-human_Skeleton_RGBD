import numpy as np
from numpy.linalg import multi_dot

#Constant
N_STATES=9      # Number of states: [x,y,z,xprime,yprime,zprime,xpp,ypp,zpp]
N_MEASURE=3     # Number of measurements: [x,y,z]
I = np.identity(N_STATES)


class KalmanFilter():
    def __init__(self, freq: float = 30.0):
        """ Class Builder """

        self.dt = 1.0/freq                          # 1 / SampleFrequency
        self.t = None                               # time, useless

        # Velocity Model matrix   x1 = x0 + vt + a*t^2/2
        # x_hat = A x_t-1

        self.A = np.identity(N_STATES)
        self.A[0:3,3:6] = np.identity(3)*self.dt
        self.A[3:6,6:np.size(self.A,1)] = np.identity(3)*self.dt
        self.A[0:3,6:np.size(self.A,1)] = np.identity(3)*(self.dt**2)*0.5
        self.A[6:np.size(self.A,0),6:np.size(self.A,1)] = np.identity(3)

        self.C = np.zeros((N_MEASURE, N_STATES))
        self.C[0:3,0:3] = np.identity(3)

        q =  np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1])/90
        self.Q = np.identity(N_STATES)*q                        # processing Noise
        self.R=np.identity(N_MEASURE)*[0.05, 0.05, 0.1]/90      # measurement Noise

        self.Sigma : np.ndarray = None           # Covariance A

        self.x_hat = None       # x_t-1
        self.x_hat_new = None   # x_t
        self.initialized = False

        # self.y = None           # I would like to add also y : x,y,z filtrati C*x_hat_new
        self.skip_measure = 0

    def initialize(self, keypoints_pos) -> None:
        """
        Method for initialize the Kalman
        @ keypoints_pos: [D(x,y,z),]
        """

        #Initial settings
        self.t = 0.0                                # time, useless
        # self.Sigma = np.ones((N_STATES,N_STATES))*1e-1      # position state
        self.Sigma = np.identity(N_STATES)          # init Covariance Mat


        #First state
        self.x_hat = np.array([keypoints_pos[0],keypoints_pos[1],keypoints_pos[2], 0,0,0, 0,0,0])    # x,y,z like measured, vel and acc = 0
        self.x_hat_new = self.x_hat

        #Set initialized flag
        self.initialized = True
        self.skip_measure = 0


    def getMeasAfterInitialize(self):
        """ x -> measurement without noise
        """
        return self.C.dot(self.x_hat)

    def update_state(self):
        self.x_hat = self.x_hat_new
        self.t += self.dt

    def updateOpenLoop(self) -> None:
        """
        Method for update without Kalman but only based on model
        return measurement
        """
        self.x_hat_new = self.A.dot(self.x_hat)
        self.Sigma = multi_dot( [self.A , self.Sigma , self.A.T ]) + self.Q
        # not sure: Kalman for Intermittent Observation : https://arxiv.org/pdf/0903.2890.pdf, https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1333199

        # Update x_mu and t
        self.update_state()

        # return measurement: vector <x,y,z>
        return self.C.dot(self.x_hat_new)
    

    def update(self,keypoints_pos):
        """
        Method for update Kalman estimation
        @ keypoints_pos: [D(x,y,z),]
        """
        # State a-priori estimation
        self.x_hat_new = self.A.dot(self.x_hat)
        self.meas_new = self.C.dot(self.x_hat_new)
        
        # check precision on depth-axis of the Kalman Filter
        #             large depth diff
        if abs(self.meas_new[2]-keypoints_pos[2]) > 0.5 and self.skip_measure < 3 and self.initialized:
            self.skip_measure += 1
            return self.updateOpenLoop()
        
        elif self.skip_measure >= 3:                # too much failure
            self.initialize(keypoints_pos)          # re-init
            return self.getMeasAfterInitialize()
        
        else:
            self.Sigma = multi_dot( [self.A , self.Sigma , self.A.T ]) + self.Q

            # Rt = self.R se  keypoints_pos - self.C.dot(self.x_hat_new)  piccolo, oppure self.Rlost
            
            # TODO: solve singularity error
            try:
                K = multi_dot([self.Sigma, self.C.T, np.linalg.inv( multi_dot([self.C , self.Sigma, self.C.T]) + self.R) ])
            except np.linalg.LinAlgError as e:
                raise

            # A-posteriori correction
            self.x_hat_new += K.dot( keypoints_pos - self.meas_new )    # new mu vector
            self.Sigma = (I - K.dot( self.C )).dot(self.Sigma)          # new Sigma

            self.meas_new = self.C.dot(self.x_hat_new)     # Adding by me      measurement
            # Update
            self.x_hat = self.x_hat_new
            self.t +=self.dt

            self.skip_measure=0
            return self.meas_new

    def getCartesianVelocity(self):
        """
        Getter method for giving the keypoint cartesian velocity (vx, vy, vz)
        @ return: [vx,vy,vz]
        """
        return self.x_hat_new[3:6]
    
    def getCartesianAcceleration(self):
        """
        Getter method for giving keypoint cartesian acceleration (ax, ay, az)
        @ return: [ax,ay,az]
        """
        return self.x_hat_new[6:]

    def getCovariance(self):
        return np.diagonal(self.Sigma)

    def getPosDevSt(self):
        """
        Getter method for giving keypoint standard deviation of keypoint position
        @ return: [dev_st x, dev_st y, dev_st z]
        """
        return np.sqrt(np.diagonal(self.Sigma))[0:3]
    
    def getVelDevSt(self):
        """
        Getter method for giving keypoint standard deviation of keypoint velocity
        @ return: [dev_st vx, dev_st vy, dev_st vz]
        """
        return np.sqrt(np.diagonal(self.Sigma))[3:6]
    
    def getAccDevSt(self):
        """
        Getter method for giving keypoint standard deviation of keypoint acceleration
        @ return: [dev_st ax, dev_st ay, dev_st az]
        """
        return np.sqrt(np.diagonal(self.Sigma))[6:]

    def reset(self):
        """
        Method for reset the kalman filter
        """
        self.x_hat = np.zeros((N_STATES,))
        self.t = 0
        self.initialized=False
