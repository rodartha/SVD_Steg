import os
import numpy
import imageio
import math
import random
import binascii
import pprint
from numpy.linalg import solve
from numpy import dot

block_size = 4
cols_protected = 1

testCol1 = [ 0.50324552,  0.50206162,  0.50148479, -0.49314408]

testCol2 = [0.37595182, 0.37715537, 0.07426397, 0.84314822]

testCol3 = [ -0.49106614,  -0.13807735, -0.83477968,  0.20719936]

testCol4 = [0.60353809, -0.76591292, 0.21480399, 0.05457491]
'''
testMatrix =    numpy.array([[ 0.50324552,  0.50206162,  0.50148479, -0.49314408],
                 [0.37595182, 0.37715537, 0.07426397, 0.84314822],
                 [ -0.49106614,  -0.13807735, -0.83477968,  0.20719936],
                 [0.60353809, -0.76591292, 0.21480399, 0.05457491]])'''

# after embedding
testMatrix =     numpy.array([[ 0.50324552, -0.37595182, -0.49106614,  0.60353809],
                            [ 0.50206162,  0.37715537,  -0.13807735, -0.76591292],
                            [ 0.50148479,  0.07426397, -0.83477968,  0.21480399],
                            [-0.49314408,  0.84314822,  0.20719936,  0.05457491]])

'''
print(testCol1)
print()
print(testCol2)
print()
print("dot product")
print(dot(testCol1, testCol2))
print()
print("TESTING")
print()


testCol2[0] = math.fabs(testCol2[0])*-1
testCol2[1] = math.fabs(testCol2[1])*1
testCol2[2] = math.fabs(testCol2[2])*1
print(testCol1)
print()
print(testCol2)
print()
print("dot product")
print(dot(testCol1, testCol2))
print()'''

print()
print("original matrix:")
print(testMatrix)
print()

coeff = [[]]
sol = [[]]

orthog_bits = 1
for i in range(cols_protected, block_size-1):

    if orthog_bits == 1:
        coeff = testMatrix[block_size-1, i-1]
        sol = -dot(testMatrix[0:block_size-1, i-1], testMatrix[0:block_size - 1, i])
    else:
        coeff = numpy.zeros((orthog_bits, orthog_bits))
        sol = numpy.zeros((orthog_bits, 1))

        for j in range(0, i):
            #print("j: " + str(j))

            for k in range(block_size-orthog_bits, block_size):
                #print("k: " + str(k))
                coeff[j, k-block_size+orthog_bits] = testMatrix[k, j]
                #print(coeff)

            sol[j][0] = -dot(testMatrix[0:block_size-orthog_bits, j], testMatrix[0:block_size - orthog_bits, i])

            '''
    print("coefficient matrix")
    print()
    print(coeff)
    print()
    print("solution matirx:")
    print()
    print(sol)
    print()
    #print(sol)'''

    if orthog_bits > 1:
        res = solve(coeff, sol)
        #print("res: ")
        #print(res)

        res = res.ravel()
        testMatrix[block_size-orthog_bits:block_size, i] = res

        print("after embedding cycle: " + str(i))
        print(testMatrix)

        print("testing dot products")
        for g in range(0, i):
            dotprod = -dot(testMatrix[0:block_size, g], testMatrix[0:block_size, i])
            print(dotprod)
            assert(math.fabs(dotprod) < .000001)

    else:
        res = sol/coeff
        #print("res: ")
        #print(res)
        #print()
        testMatrix[block_size-1, i] = res

        print("after embedding cycle: " + str(i))
        print(testMatrix)

        print("testing dot products")
        for g in range(0, i):
            dotprod = -dot(testMatrix[0:block_size, g], testMatrix[0:block_size, i])
            print(dotprod)
            assert(math.fabs(dotprod) < .000001)

    print()

    orthog_bits += 1


print("final:")
print(testMatrix)

'''
print("TESTING HARDCODED")
print()
coeff = testCol1[3]

#print(testCol1[0:3])
#print(testCol2[0:3])
#print()
sol = -dot(testCol1[0:3],testCol2[0:3])
print("sol: ")
print (sol)

#res = solve(coeff, sol)
res = sol/coeff
print("res: ")
print(res)

testCol2[3] = res
print()
print("dot product adjusted u1 and u2")
print(dot(testCol1, testCol2))
print()

coeff = numpy.zeros((2,2))
sol = numpy.zeros((2,1))

coeff[0][0] = testCol1[2]
coeff[0][1] = testCol1[3]
coeff[1][0] = testCol2[2]
coeff[1][1] = testCol2[3]

sol[0][0] = -dot(testCol1[0:2],testCol3[0:2])
sol[1][0] = -dot(testCol2[0:2],testCol3[0:2])

print("sol:")
print(sol)
print()

res = solve(coeff,sol)
print("res")
print(res)

testCol3[2:4] = res

print()

print("dot product adjusted u1 and u3")
print(dot(testCol1, testCol3))
print()
print("dot product adjusted u2 and u3")
print(dot(testCol2, testCol3))
print()
'''
