from streaming import Streaming

stream = Streaming(size = [424, 240], scope = [100, 3000], minChange = 50, nb_pixels_max = 8000)
stream.start()