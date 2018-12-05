"""Steganographer Class."""
import os
import numpy
import imageio
import math
import random
import binascii
import pprint
from numpy.linalg import solve
from numpy import dot
from svd_steg.helper import progress_bar


class Steganographer:
    """Class handles embedding and decoding."""

    def __init__(self, method, image_in,
                 image_file, message_in=None, message_file=None):
        """Initialize Variables."""

        print("INITIALIZING Steganographer with method " + method)

        self.message = message_in
        self.message_file = message_file
        self.image = image_in.astype(numpy.int32)
        self.image_file = image_file
        self.method = method
        self.embedded_image = self.image
        self.error_count = 0
        self.error_flag = 0

    def output_embedded_image(self):
        """Ouput an embedded image as IMAGENAME_steg."""
        file_split = self.image_file.split('.')
        output_filename = file_split[0] + '_steg.' + file_split[1]

        file_ending = file_split[1]
        embedded_image = (self.embedded_image).astype(numpy.uint8)
        if file_ending == 'jpg':
            embedded_image = numpy.zeros([self.embedded_image.shape[0], self.embedded_image.shape[1], 3])
            for i in range(0, 3):
                embedded_image[:,:,i] = self.embedded_image

        imageio.imwrite("output/" + output_filename, embedded_image)

    def output_decoded_text(self):
        """Output decoded text as IMAGENAME_text.txt."""
        file_split = self.image_file.split('.')
        output_filename = file_split[0] + '_text.txt'

        file = open("ouput/" + output_filename, 'w+')
        file.write(self.message)
        file.close()

    def output(self):
        """Determine what file to output."""
        if not os.path.exists("output"):
            os.makedirs("output")
        if self.method == "embed":
            self.output_embedded_image()
        else:
            self.output_decoded_text()

    def create_block_image(self, block_size):
        """ Create blocks """
        return None

    def binarize_message(self):
        """ Turn string into binary """
        binary_message = ''.join(format(ord(x), 'b') for x in self.message)
        binary_list = []
        for character in binary_message:
            if character == '0':
                binary_list.append(-1)
            else:
                binary_list.append(1)

        return binary_list

    def computeSVD(self, image_block):
        """compute the SVD of a single image block (will add input later)"""

        #print(image_block)
        U, S, VT = numpy.linalg.svd(image_block)

        # create blank m x n matrix
        Sigma = numpy.zeros((U.shape[1], VT.shape[0]))

        # populates Sigma by transplating s matrix into main diaganol
        #Sigma[:VT.shape[0], :VT.shape[0]] = numpy.diag(numpy.diag(s))

        for i in range(0, S.shape[0]):
            for j in range(0, S.shape[0]):
                if i == j:
                    Sigma[i,j] = S[i]

        '''
        print("SIGMA")
        print(Sigma)
        '''
        block = U.dot(Sigma.dot(VT))
        '''
        print("reconstructed block")
        print(block)
        print()
        '''
        return [U, Sigma, VT]


    def make_column_orthogonal(self, U_matrix, cols_protected, col, orthog_bits):

        # if the col is protected it is already orthogonal
        if col < cols_protected:
            return

        # get block size
        block_size = U_matrix.shape[0]

        # set this matrix = block with embedded bits
        testMatrix = U_matrix

        # empty for the results
        coeff = [[]]
        sol = [[]]

        # handle case that we are on the first column, we simply let coeff the last element
        # in col0, and sol = the negative dot product of the entries in col0 and col1 up until
        # the last entry
        if orthog_bits == 1:
            coeff = testMatrix[block_size-1, col-1]
            sol = -dot(testMatrix[0:block_size-1, col-1], testMatrix[0:block_size - 1, col])
        else:

            # else coeff = the [orthog_bits*orthog_bits] matrix consisting of
            # the bottom orthog_bits elements from each column 0...i (non inclusive)
            # essentially taking all columns previous to its [unkown*unkown] bottom values
            coeff = numpy.zeros((orthog_bits, orthog_bits))
            sol = numpy.zeros((orthog_bits, 1))

            print("num orthog bits: " + str(orthog_bits))

            # actually transplant the values into our coeff and sol matrices
            # this is just actually taking hte values from testMatrix and placing
            # them in our matrices as described above

            # the rationale is that if our unkowns * the knowns in the left vectors = the -dot product of the knowns,
            # then their total result will be 0, this is what we are solving to accomplish
            for j in range(0, col):

                for k in range(block_size-orthog_bits, block_size):
                    coeff[j, k-block_size+orthog_bits] = testMatrix[k, j]

                sol[j][0] = -dot(testMatrix[0:block_size-orthog_bits, j], testMatrix[0:block_size - orthog_bits, col])

        '''
        print("coefficient matrix")
        print()
        print(coeff)
        print()
        print("solution matrix:")
        print()
        print(sol)
        print()
        '''

        # handle the case that that we are not on orth_bits == 1
        if orthog_bits > 1:
            try:

                # simply solve the equation
                res = solve(coeff, sol)
                #print("res: ")
                #print(res)

                # turns a matrix of matrices into a single matrix
                res = res.ravel()

                # replace the unkown values in the matrix
                testMatrix[block_size-orthog_bits:block_size, col] = res

                #print("after embedding cycle: " + str(i))
                #print(testMatrix)

                # test that all dot product are 0 and it is in fact orthogonal
                for g in range(0, col):
                    '''
                    print("testing:")
                    print(testMatrix[0:block_size, g])
                    print()
                    print("and")
                    print()
                    print(testMatrix[0:block_size, i])
                    '''

                    dotprod = -dot(testMatrix[0:block_size, g], testMatrix[0:block_size, col])
                    #print(dotprod)
                    if numpy.linalg.matrix_rank(coeff) == orthog_bits:
                        assert(math.fabs(dotprod) < .000001)
            except:
                print("could not make orthogonal 2")
                self.error_count += 1


        # handle case that we have just 1 orthogonal bit
        else:

            try:

                # as long as coeff != 0 we can easily find the single missing value given 2 columns
                if coeff != 0:

                    res = sol/coeff
                    '''
                    print("res:")
                    print(res)
                    print()
                    '''
                    testMatrix[block_size-1, col] = res
                    '''
                    print("after embedding cycle: " + str(col))
                    print(testMatrix)
                    '''
                    # again test dot products
                    for g in range(0, col):
                        '''
                        print("testing:")
                        print(testMatrix[0:block_size, g])
                        print()
                        print("and")
                        print()
                        print(testMatrix[0:block_size, i])
                        '''
                        dotprod = -dot(testMatrix[0:block_size, g], testMatrix[0:block_size, col])
                        #print(dotprod)
                        assert(math.fabs(dotprod) < .000001)
            except:

                # occasionally we get an error when the rank of our matrix_rank
                # is less than the size of it, or if we have a bunch of zeroes
                print("could not make orthogonal 1")
                self.error_count += 1

        return testMatrix


    def embed(self):
        """Embed message into an image."""

        #A = []

        # Set block size
        block_size = 4
        cols_protected = 1 # currently only works with 1, would be great to get 2 working

        num_rows = ((self.embedded_image).shape)[0]
        num_cols = ((self.embedded_image).shape)[1]

        # bits per block
        # hardcoded for now for an 4x4 with 1 column protected matrix, math wasnt working quite right
        bpb = 5

        # num blocks possible
        num_blocks = (num_rows * num_cols)/(block_size * block_size)

        img_cap = math.floor(bpb*num_blocks)

        # take this with a grain of salt for now, not sure if accurate
        print("MAX IMAGE HIDING CAPACITY: " + str(img_cap))
        print()

        # calculate the maximum number of blocks per row/col
        row_lim = math.floor(num_rows/block_size)
        print("row_lim: " + str(row_lim))
        col_lim = math.floor(num_cols/block_size)
        print("col_lim: " + str(col_lim))

        # convert message to bits to be embeded (currenty only supports block size 8)
        binary_message = self.binarize_message()

        # looping through each block
        for j in range(col_lim):
            for i in range(row_lim):

                # isolate the block
                block = self.embedded_image[block_size*i:block_size*(i+1), j*block_size:block_size*(j+1)]

                print("original block: ")
                print(block)
                print()

                # compute the SVD
                U, S, VT = self.computeSVD(block)



                # rememeber that A = U*Sigma*VT can be made standard using
                # a matrix that is the identity martix with +-1 as the diaganol values
                # giving use A = U*Sigma*V^T = (U*D)*Sigma*(D*VT) because D is its own inverse
                # and by associativity, the problem is that this messes with the orthogonality

                print("original U:")
                print()
                print(U)
                print()

                for k in range(0, block_size):
                    if U[0,k] < 0:

                        # multiply entire columns by -1
                        U[0:block_size, k] *= -1
                        VT[0:block_size, k] *= -1

                print("modified U:")
                print()
                print(U)
                print()

                # prepare string for embedding (chop up binary)
                to_embed = ""
                if len(binary_message) >= bpb:
                    to_embed = binary_message[0:bpb]
                    binary_message = binary_message[bpb:]
                else:
                    to_embed = binary_message
                    binary_message = ""

                # for testing
                if to_embed == "":
                    break

                print("EMBEDDING: ")
                print(to_embed)
                print()

                # for the embedding
                U_mk = U

                # not sure why they do this, but its in the psuedocode (not sure if it is working)
                S_Prime = S

                avg_dist = (S[1,1] + S[block_size-1,block_size-1])/(block_size-2);
                for k in range (2, block_size):
                    S_Prime[k,k]= S[1,1] - (k-2)*avg_dist


                print("U-Matrix before embedding: ")
                print(U_mk)
                print()

                 # m is columns, n is rows:
                num_orthog_bits = 1
                message_index = 0
                for m in range(0, block_size):
                    for n in range(0, block_size):

                        # only embed as long as the message still has bits to embed
                        if message_index < len(to_embed):

                            # if last column, dont embed but make orthogonal
                            if (m == block_size-1):
                                pass

                            # protect column
                            elif m < cols_protected:
                                    U_mk[n, m] = U[n, m]

                            # embed bits
                            elif n < (block_size - num_orthog_bits):
                                U_mk[n,m] = to_embed[message_index] * math.fabs(U[n,m])
                                message_index += 1

                    # if we are past protected cols then make the current column orthogonal to the previos ones
                    if m >= cols_protected:
                        U_mk = self.make_column_orthogonal(U_mk, cols_protected, m, num_orthog_bits)
                        num_orthog_bits += 1


                print("U-Matrix after embedding: ")
                print(U_mk)
                print()


                # normalize the new U matrix by dividng by the magnitude of each column (not sure why)
                for x in range(0, block_size):
                    norm_factor = math.sqrt(dot(U_mk[0:block_size, x],U_mk[0:block_size, x]))
                    for y in range(0, block_size):
                        U_mk[y,x] /= norm_factor


                # round result to be whole numbers
                block = numpy.round(U_mk.dot(S_Prime.dot(VT)))


                # wasn't quite working, its to make sure all pixels within 0-255
                # but they are using a tiff which might have different properties?
                '''
                for x in range(0, block_size):
                    for y in range(0, block_size):
                        if block[x, y] > 255:
                            block[x, y] = 255
                        if block[x, y] < 0:
                            block[x, y] = 0
                '''

                block = block.astype(numpy.uint8)
                print("reconstructed block")
                print(block)
                print()

                # reassign the block after modification
                self.embedded_image[block_size*i:block_size*(i+1), j*block_size:block_size*(j+1)] = block

        return None

    def decode(self):
        """Decode message from image."""
        # TODO: Implement
        return None

    def format_image(self):
        file_split = self.image_file.split('.')
        file_ending = file_split[1]
        if file_ending == 'jpg':
            self.image = self.image[:,:,0]
        self.embedded_image = self.image

    def run(self):
        """Run Steganography class."""
        self.format_image()
        if self.method == "embed":
            print("RUNNING steganographer with METHOD embed")
            print()
            print("Image Dimensions: " + str((self.image).shape))
            print()
            #print(self.image)
            #print()
            self.embed()
            print("number of errors: " + str(self.error_count))
        else:
            print("RUNNING steganographer with METHOD decode")
            print()
            self.decode()

        self.output()
