import os
import numpy
import imageio
import math
import random
import binascii
import pprint
from numpy.linalg import solve
from numpy import dot

bpb = 5
block_size = 4

temp_rcvd_msg = []

'''
original block:
[[11 10  6  2]
 [12 10  6  3]
 [12 10  6  3]
 [12 10  6  2]]

EMBEDDING:
[-1, 1, -1, -1, 1]

reconstructed block
[[11 10  6  3]
 [12 10  6  2]
 [13 10  6  3]
 [12 10  6  2]]

 original block:
[[117 119 121 141]
 [107 116 126 133]
 [107 118 135 137]
 [114 119 121 113]]

EMBEDDING:
[1, -1, -1, 1, 1]

num orthog bits: 2
num orthog bits: 3
reconstructed block
[[105 117 140 135]
 [109 125 124 125]
 [124 121 126 128]
 [106 109 111 138]]

 original block:
[[121 123 123 119]
 [123 124 125 125]
 [124 126 126 131]
 [123 107 103 105]]

EMBEDDING:
[-1, -1, 1, -1, -1]

num orthog bits: 2
num orthog bits: 3
reconstructed block
[[127 127 118 114]
 [134 119 123 121]
 [118 129 133 127]
 [111 106 102 120]]
'''


recovered = numpy.array([[11, 10,  6,  3],
                        [12, 10,  6,  2],
                        [13, 10,  6,  3],
                        [12, 10,  6,  2]])

embedding = [0, 1, 0, 0, 1]

'''
recovered = numpy.array([[105, 117, 140, 135],
                        [109, 125, 124, 125],
                        [124, 121, 126, 128],
                        [106, 109, 111, 138]])

embedding = [1, 0, 0, 1, 1]

recovered = numpy.array([[127, 127, 118, 114],
                         [134, 119, 123, 121],
                         [118, 129, 133, 127],
                         [111, 106, 102, 120]])
embedding = [0, 0, 1, 0, 0]
'''

U, S, VT = numpy.linalg.svd(recovered)

print("U:")
print(U)
print()

U_std = U

for k in range (0, block_size):
    if U[0,k] < 0:
        U_std[0:block_size, k] = -1*U[0:block_size,k];


print("U_std:")
print(U_std)
print()

next_spot = 0;


for k in range(1, block_size-1):

    for p in range(0, block_size-k):
        if U_std[p, k] < 0:
            temp_rcvd_msg.append(0);
        else:
            temp_rcvd_msg.append(1);

print("embedded:")
print(embedding)
print()
print("recovered: ")
print()
print(temp_rcvd_msg)

    # if the entry being examined is < 0 then the bit
    # is a 0 otherwise the bit is a 1
















#space saver
