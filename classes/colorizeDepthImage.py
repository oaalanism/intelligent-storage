import numpy as np

class ColorizeDepthImage:

    """
    Colorizer Object has the task to apply depth image compression by colorization from : 
    --------------------https://dev.intelrealsense.com/docs/depth-image-compression-by-colorization-for-intel-realsense-depth-cameras----------------------------- 
    """

    def __init__(self):
        pass

    def buildRedPixel(self, d_normal):
        """
        Red image component created from the depth data 

        Parameters
        ----------
            d_normal : Int 
                normalized depth frame

        Returns
        -------
            pr : Array
                Red array for the image
        
        """

        filtre1 = np.where(np.bitwise_or(np.bitwise_and(0 <= d_normal, d_normal <= 255), np.bitwise_and(1275 <= d_normal, d_normal <= 1529)), 255, 0)
        filtre2 = np.where(np.bitwise_and(255 < d_normal, d_normal <= 510), 255 - d_normal, 0)
        filtre3 = np.where(np.bitwise_and(510 < d_normal, d_normal <= 1020), 0, 0)
        filtre4 = np.where(np.bitwise_and(1020 < d_normal, d_normal <= 1275), d_normal - 1020, 0)
        pr = filtre1 + filtre2 + filtre3 + filtre4
        return pr

    def buildGreenPixel(self, d_normal):
        """
        Green image component created from the depth data 

        Parameters
        ----------
            d_normal : Int 
                normalized depth frame

        Returns
        -------
            pg : Array
                Green array for the image
        
        """
        filtre1 = np.where(np.bitwise_and(0 < d_normal, d_normal <= 255), d_normal, 0)
        filtre2 = np.where(np.bitwise_and(255 < d_normal, d_normal <= 510), 255, 0)
        filtre3 = np.where(np.bitwise_and(510 <d_normal, d_normal <= 765), 765 - d_normal, 0)
        filtre4 = np.where(np.bitwise_and(765 < d_normal, d_normal <= 1529), 0, 0)
        pg = filtre1 + filtre2 + filtre3 + filtre4
        return pg

    def buildBluePixel(self, d_normal):
        """
        Blue image component created from the depth data 

        Parameters
        ----------
            d_normal : Int 
                normalized depth frame

        Returns
        -------
            pb : Array
                Blue array for the image
        
        """
        filtre1 = np.where(np.bitwise_and(0 < d_normal, d_normal <= 765), 0, 0)
        filtre2 = np.where(np.bitwise_and(765 < d_normal, d_normal <= 1020), d_normal - 765, 0)
        filtre3 = np.where(np.bitwise_and(1020 < d_normal, d_normal <= 1275), 255, 0)
        filtre4 = np.where(np.bitwise_and(1275 < d_normal, d_normal <= 1529), 1275 - d_normal, 0)
        pb = filtre1 + filtre2 + filtre3 + filtre4
        return pb

    def buildImage(self, d_normal):
        """
        Function to create colorized image

        Parameters
        ----------
            d_normal : Array
                depth normalized image
        Returns
        -------
            image : Array
                3 dimentional array represented colorized image
        """
        pr = self.buildRedPixel(d_normal)
        pg = self.buildGreenPixel(d_normal)
        pb = self.buildBluePixel(d_normal)

        image = np.zeros((d_normal.shape[0], d_normal.shape[1], 3))
        image[:,:,2] = pr
        image[:,:,1] = pg
        image[:,:,0] = pb
        image = image.astype('uint8')
        return image

    def appplyColorization(self, d, d_min, d_max):
        """
        Apply colorization
        Parameters
        ----------
            d : Array
                depth image
            d_min : Int
                Minimum distance in the scene
            d_max : Int
                Maximum distance in the scene
        Returns
        -------
            depth_color_image : Array
                3 dimentional array represented colorized image
        """
        d_normal = ((d - d_min) / (d_max - d_min)) * 1529
        self.depth_color_image = self.buildImage(d_normal)
        return self.depth_color_image

    