import numpy as np

class ColorizeDepthImage:

    def buildRedPixel(self, d_normal):
        filtre1 = np.where(np.bitwise_or(np.bitwise_and(0 <= d_normal, d_normal <= 255), np.bitwise_and(1275 <= d_normal, d_normal <= 1529)), 255, 0)
        filtre2 = np.where(np.bitwise_and(255 < d_normal, d_normal <= 510), 255 - d_normal, 0)
        filtre3 = np.where(np.bitwise_and(510 < d_normal, d_normal <= 1020), 0, 0)
        filtre4 = np.where(np.bitwise_and(1020 < d_normal, d_normal <= 1275), d_normal - 1020, 0)
        pr = filtre1 + filtre2 + filtre3 + filtre4
        return pr

    def buildGreenPixel(self, d_normal):
        filtre1 = np.where(np.bitwise_and(0 < d_normal, d_normal <= 255), d_normal, 0)
        filtre2 = np.where(np.bitwise_and(255 < d_normal, d_normal <= 510), 255, 0)
        filtre3 = np.where(np.bitwise_and(510 <d_normal, d_normal <= 765), 765 - d_normal, 0)
        filtre4 = np.where(np.bitwise_and(765 < d_normal, d_normal <= 1529), 0, 0)
        pg = filtre1 + filtre2 + filtre3 + filtre4
        return pg

    def buildBluePixel(self, d_normal):
        filtre1 = np.where(np.bitwise_and(0 < d_normal, d_normal <= 765), 0, 0)
        filtre2 = np.where(np.bitwise_and(765 < d_normal, d_normal <= 1020), d_normal - 765, 0)
        filtre3 = np.where(np.bitwise_and(1020 < d_normal, d_normal <= 1275), 255, 0)
        filtre4 = np.where(np.bitwise_and(1275 < d_normal, d_normal <= 1529), 1275 - d_normal, 0)
        pb = filtre1 + filtre2 + filtre3 + filtre4
        return pb

    def buildImage(self, d_normal):
        pr = self.buildRedPixel(d_normal)
        pg = self.buildGreenPixel(d_normal)
        pb = self.buildBluePixel(d_normal)

        image = np.zeros((d_normal.shape[0], d_normal.shape[1], 3))
        image[:,:,2] = pr
        image[:,:,1] = pg
        image[:,:,0] = pb
        image = image.astype('uint8')
        return image

    def appplyColorization(self, frame, d_min, d_max):

        d_normal = (frame / d_max) * 1529

        self.depth_color_image = self.buildImage(d_normal)
        return self.depth_color_image

    def __init__(self):
        pass