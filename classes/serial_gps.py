import serial


def ret_coords():
    SERIAL_PORT = "/dev/ttyUSB1"
    running = True

    def formatDegreesMinutes(coordinates, digits):
        parts = coordinates.split(".")

        if (len(parts) != 2):
            return coordinates

        if (digits > 3 or digits < 2):
            return coordinates
        
        left, right = parts[0], parts[1]
        degrees = str(left[:digits])
        minutes = str(left[digits:]+"."+right[:6])
        degrees, minutes = float(degrees), float(minutes)
        coords = degrees+ (minutes/60)
        return (str(coords))

    # This method reads the data from the serial port, the GPS dongle is attached to,
    # and then parses the NMEA messages it transmits.
    # gps is the serial port, that's used to communicate with the GPS adapter
    def getPositionData(gps):
        if gps.readline().decode("utf-8")=="":
            return (999,999)
        checkNMEA= True
        while (checkNMEA == True):
            data = gps.readline().decode("utf-8")
            message = data[0:6]
            if (message == "$GPRMC"):
                checkNMEA= False
            
       
        if (message == "$GPRMC"):
            # GPRMC = Recommended minimum specific GPS/Transit data
            # Reading the GPS fix data is an alternative approach that also works
            parts = data.split(",")
            #print(parts)
            if parts[2] == 'V':
                return (999,999)
            else:
                # Get the position data that was transmitted with the GPRMC message
                # In this example, I'm only interested in the longitude and latitude
                # for other values, that can be read, refer to: http://aprs.gids.nl/nmea/#rmc
                longitude = "-" + formatDegreesMinutes(parts[5], 3) #if switch_east==False else -formatDegreesMinutes(parts[4], 3)
                latitude = formatDegreesMinutes(parts[3], 2)
                speed = parts[7]
                return(longitude, latitude)

                
    
        else:
            # Handle other NMEA messages and unsupported strings
            pass
        

    #print ("Application started!")
    gps = serial.Serial(SERIAL_PORT, baudrate = 9600, timeout = 0.5)

   
    try:
        getPositionData(gps)
    except KeyboardInterrupt:
        running = False
        gps.close()
        #print ("Application closed!")
    except:
        # You should do some error handling here...
        print ("Application error!")
    return(getPositionData(gps))
