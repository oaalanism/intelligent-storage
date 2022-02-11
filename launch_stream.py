from classes.streaming import Streaming


"""
Summary:
This script launch streaming for depth camera D455 to save data in three representations:
----------------Raw Data-----------------------------------------
----------------Depth Frames colorized store in videos-----------
----------------Sparce Matrix -----------------------------------

Streaming object will automatic create all necessary repositories
"""


size = [424, 240]

scope = [100, 2500]

minChange = 100

nb_pixels_max = 50000

stream = Streaming(size, scope, minChange, nb_pixels_max)