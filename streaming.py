import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import time
from storage import Storage

class Streaming: 

    def showFrame(self, depth_color_image):
        cv2.imshow("Depth Stream", depth_color_image)
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            
    

    def start(self):
        try:
            self.pipeline.start(self.config)
            nb_frame = 0

            cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
            colorizer = rs.colorizer()

            begin = time.time()
            end = begin
            current = int(end - begin)
            while current <= 60:
                if(int(end - begin) - current >= 1 ):
                    current = int(end - begin)
                    print("Time: " + str(int(end - begin)))
                frames = self.pipeline.wait_for_frames()

                depth = frames.get_depth_frame() # composite_frame
            
                
                if not depth: continue
                else:
                    self.storage.setFrame(np.asanyarray(depth.get_data(), dtype=np.int32))

                    self.storage.store(current, nb_frame)
                        
                    depth_color_frame = colorizer.colorize(depth)
                    depth_color_image = np.asanyarray(depth_color_frame.get_data())

                    self.showFrame(depth_color_image)

                    nb_frame = nb_frame + 1

                end = time.time()
        except Exception as e:
            print(e)
            pass


    def __init__(self, size, scope, minChange):
        """
            size: width, height
            scope: minSocpe, maxScope in mm
            min change in mm
        """

        self.verboseA = False
        self.width = size[0]
        self.height = size[1]

        ctx = rs.context()
    
        listDevices = ctx.query_devices()
        if listDevices == 0:
            print("No device detected")
        else:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 15)
            self.storage = Storage([self.width, self.height], 60000, scope, minChange)
            
            self.parameters = [False, '7', '5', False, 500.0, 20000.0, 2000.0, 10000000.0, True, False, 0.7, 0.0, True, True, 0.3, 0.3, True, True, 0.45, 0.2, '30', ([0, 0],[212, 0],[212, 90], [180, 50], [40, 50],[0, 90]), ([0, 100], [40, 60], [172, 60], [212, 100], [212, 120], [0, 120]), ([-5,-5],[216,-5],[216,50],[-5,50]), ([-5,51],[216,51],[216,125],[-5,125]), (20,20,0), (35,30,0), '40']
        
