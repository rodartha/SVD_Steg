"""Steganographer Class."""
import os
import numpy
import imageio
import math
import random
import binascii
import pprint

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

    def output_embedded_image(self):
        """Ouput an embedded image as IMAGENAME_steg."""
        file_split = self.image_file.split('.')
        output_filename = file_split[0] + '_steg.' + file_split[1]

        file_ending = file_split[1]
        embedded_image = self.embedded_image
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
        U, s, VT = numpy.linalg.svd(image_block)

        # create blank m x n matrix
        Sigma = numpy.zeros((U.shape[1], VT.shape[0]))

        # populates Sigma by transplating s matrix into main diaganol
        Sigma[:VT.shape[0], :VT.shape[0]] = numpy.diag(numpy.diag(s))

        '''
        print("SIGMA")
        print(Sigma)
        '''

        return [U, Sigma, VT]

    def embed(self):
        """Embed message into an image."""

        #A = []

        # Set block size
        block_size = 8
        cols_protected = 2

        num_rows = ((self.embedded_image).shape)[0]
        num_cols = ((self.embedded_image).shape)[1]

        # bits per block
        bpb = math.floor(((block_size - cols_protected-1)*(block_size - cols_protected))/2)

        # num blocks possible
        num_blocks = (num_rows * num_cols)/(block_size * block_size)

        img_cap = math.floor(bpb*num_blocks)

        # take this with a grain of salt for now, not sure if accurate
        print("MAX IMAGE HIDING CAPACITY: " + str(img_cap))
        print()

        '''
        GENERATE A SEQUENTIAL MATRIX block_size*block_size FOR TESTING

        for i in range(block_size*3):
            A.append(numpy.arange(block_size*i*3, block_size*(i+1)*3))

        A = numpy.array(A)

        self.embedded_image = A


        print(A)
        '''

        # calculate the maximum number of blocks per row/col
        row_lim = math.floor(num_rows/block_size)
        print("row_lim: " + str(row_lim))
        col_lim = math.floor(num_cols/block_size)
        print("col_lim: " + str(col_lim))

        # convert message to bits to be embeded (currenty only supports block size 8)
        binary_message = self.binarize_message()
        # print(binary_message)
        # print()

        # looping through each block
        for j in range(col_lim):
            for i in range(row_lim):

                # isolate the block
                block = self.embedded_image[block_size*i:block_size*(i+1), j*block_size:block_size*(j+1)]

                '''
                TO TEST THE BLOCKING WITH RANDOM MODIFICATIONS OF EACH BLOCK

                print(block)
                print()

                block *= random.randint(1,5)
                block %= 256

                print(block)
                print()
                '''

                # compute the SVD
                U, S, VT = self.computeSVD(block)

                U_std = U
                VT_prime = VT

                # rememeber that A = U*Sigma*VT can be made standard using
                # a matrix that is the identity martix with +-1 as the diaganol values
                # giving use A = U*Sigma*V^T = (U*D)*Sigma*(D*VT) because D is its own inverse
                # and by associativity thus

                for k in range(0, block_size):
                    if U[0,k] < 0:

                        # multiply entire columns by -1
                        U_std[0:(block_size-1),k] *= -1
                        VT_prime[0:(block_size-1),k] *= -1

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
                print(block)
                print()

                U_mk = U_std

                '''
                print("U-Matrix before embedding: ")
                print(U_mk)
                print()
                '''

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
                                    U_mk[n][m] = U_std[n][m]

                            # embed bits
                            elif n < (block_size - num_orthog_bits):
                                U_mk[n][m] = to_embed[message_index] * math.fabs(U_std[n][m])
                                message_index += 1

                            # make orthogonal
                            else:
                                # TODO: function that creates orthogonal values
                                pass


                    num_orthog_bits += 1


                print("U-Matrix after embedding: ")
                print(U_mk)
                print()

                # need to figure out orthogonality stuff

                '''  IF YOU WANT TO TEST THE DOT PRODUCT OF TWO COLS
                testCol1 = U_mk[0:block_size, 0]
                testCol2 = U_mk[0:block_size, 1]
                print(testCol1)
                print()
                print(testCol2)
                print()
                print("dot product")
                print(dot(testCol1, testCol2))
                print()
                '''

                # normalize

                # reconstruction

                #block = U_mk.dot(S.dot(VT_prime))

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
        else:
            print("RUNNING steganographer with METHOD decode")
            print()
            self.decode()

        self.output()
