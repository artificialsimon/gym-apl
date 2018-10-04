#!/usr/bin/python

""" Imports all the terrain from the 500x500 grind to a dictionary
	env[position_x, position_y] = [latitude(y), longitude(x), altitude, name,
	    description]
	An running version with a loaded map of APL should be reachable
	Runs on python 2.7
	Saves the dictionary to 'environment_dump.pkl' using Pickle,
		it will overwrite any existing file with that name
"""

import socket
import sys
import csv
import pickle

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = ('0.0.0.0', 14855)
env = {}
for x in range(0, 500):
    for y in range(0, 500):
        message = "('FLIGHT', 'MS_QUERY_TERRAIN', {}, {})".format(x, y)
        sent = sock.sendto(message, server_address)
        data, server = sock.recvfrom(4096)
        for line in csv.reader([data]):# delimiter=',', quotechar='"'):
		env[x, y] = [int(line[2]), int(line[3]), int(line[4]), line[5][1:].replace("'", ''), line[6][1:-1].replace("'", '')]
#print env
sock.close()
with open("environment_dump.pkl", "wb") as file_p:   # Pickling
	pickle.dump(env, file_p)
	file_p.close()
