import pyrealsense2 as rs
import numpy as np
import cv2
import time
from classes.storage import Storage

class Streaming: 

    """
    The streaming object has the objectif of stream data for 60 senconds from camera Intel RealSense D455 

    Camera might be connected by an USB, otherwise stream will return "No device detected"

    This object use Storage object to store depth data  

    Parameters
    ----------

    size : Array 
        This arguments is the dimention of the image, this information is necessary for others objects.
        One dimention array with two variables, first variable is the width and second is the lenght of an image

    scope : Array
        Scope is a parameter necessary to set the scope depth data to store
        One dimention array with two variables, first argument set min distance and second max distance

    min_change : Int
        Set smallest difference between two pixels to consider that they have changed
        This values is defined in mm (1m ~= 1000mm)

    nb_pixels_max : Int 
        Set the minimum number of pixels changed to consider that an image is different from other.
    """

    def __init__(self, size, scope, min_change, nb_pixels_max):

        self.verboseA = False
        self.width = size[0]
        self.height = size[1]
        ctx = rs.context()
    
        listDevices = ctx.query_devices()
        if listDevices == 0:
            print("No device detected")
        else:
            self.size = size
            self.scope = scope
            self.min_change = min_change
            self.nb_pixels_max = nb_pixels_max
            self.start()
            
    def showFrame(self, depth_color_image):

        """
        colour depth camera frame display function

        Parameters
        ----------
        depth_color_image : numpy 
            3D dimentional array to represent colorized image

        Raises
        ------

        Returns
        -------

        """
        
        cv2.imshow("Depth Stream", depth_color_image)
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            
    

    def start(self):
        """
        Function to start streaming

        Parameters
        ----------

        Raises
        ------

        ImportError
            cannont import rs

        Returns
        -------
        
        """
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
            storage = Storage(self.size, self.nb_pixels_max, self.scope, self.min_change)

            pipeline.start(config)
            nb_frame = 0
            #dec_filter = rs.decimation_filter ()    
            
            colorizer = rs.colorizer()

            begin = time.time()
            end = begin
            current = int(end - begin)
            while current <= 60:
                if(int(end - begin) - current >= 1 ):
                    current = int(end - begin)
                    print("Time: " + str(int(end - begin)))
                frames = pipeline.wait_for_frames()

                depth = frames.get_depth_frame() # composite_frame
                #depth = dec_filter.process(depth)
                
                if not depth: continue
                else:
                    
                    storage.setFrame(np.asanyarray(depth.get_data(), dtype=np.int32))

                    storage.store(current, nb_frame)
                    depth_color_image = storage.depth_color_image
                    depthI = np.asanyarray(colorizer.colorize(depth).get_data())
                    depthI = cv2.cvtColor(depthI,cv2.COLOR_BGR2RGB)
                    
                    storage.depth_color_image = depth_color_image
                    storage.storeVideo()
                    self.showFrame(depthI)

                    nb_frame = nb_frame + 1

                end = time.time()

            storage.stopRecordinfVideo()
        except Exception as e:
            print(e)
            pass


    