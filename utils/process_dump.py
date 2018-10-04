#!/usr/bin/python

""" opens the dictionary with the full map of godiland and saves
    a numpy array only with the altitude of objects or objexts coded
"""
import pickle
import numpy as np

objects_code = {'mountain ridge' : 0, 'trail' : 1, 'shore bank' : 2,
        'flight tower' : 3, 'cabin' : 4, 'stripped road' : 5,
        'solo tent' : 6, 'runway' : 7, 'white Jeep' : 8,
        'water' : 9, 'pine trees' : 10, 'bush' : 11, 'active campfire ring' : 12,
        'firewatch tower' : 13, 'bushes' : 14, 'unstripped road' :15,
        'pine tree' : 16, 'blue Jeep' : 17, 'grass' : 18, 'family tent' : 19,
        'small hill' : 20, 'box canyon' : 21, 'inactive campfire ring' : 22,
        'large hill' : 23}
with open('godiland_dump.pkl', 'rb') as fp:
    q = pickle.load(fp)
    fp.close()
alt = [[0 for x in range(500)] for y in range(500)]
#a = [0 for x in range(500*500)]
for x in range(0, 500):
    for y in range(0, 500):
        #if q[x, y][2] == -1:
            #q[x, y][2] = 0
        #alt[x][y] = q[x, y][2]
        alt[x][y] = objects_code[q[x, y][3]]
        #a.append(q[x, y][3])
#print(set(a))
altn = np.array(alt, dtype=np.uint8)
with open('godiland_objects.npa', 'wb') as fp:
    np.save(fp, altn)
    fp.close()
