"""Steganographer Class."""
import os
import numpy
import imageio
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
        self.embedded_image = numpy.zeros(self.image.shape)

        print (self.method)

    def output_embedded_image(self):
        """Ouput an embedded image as IMAGENAME_steg."""
        file_split = self.image_file.split('.')
        output_filename = file_split[0] + '_steg.' + file_split[1]

        imageio.imwrite("output/" + output_filename, self.embedded_image)

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

    def computeSVD(self, image_block):
        """compute the SVD of a single image block (will add input later)"""

        U,s,VT = numpy.linalg.svd(image_block)

        # create blank m x n matrix
        Sigma = numpy.zeros((image_block.shape[0], image_block.shape[1]))
        # populate Sigma with n x n diagonal matrix
        Sigma[:image_block.shape[1], :image_block.shape[1]] = numpy.diag(s)

        '''
        print(U)
        print()
        print(Sigma)
        print()
        print(VT)
        print()
        '''

        # reconstruction
        '''
        B = U.dot(Sigma.dot(VT))

        print()
        print(B)
        '''

        return [U, Sigma, VT]

    def embed(self):
        """Embed message into an image."""

        # break image into blockSize

        # loop
            # through each block and compute svd
            # embed message bits
            # ensure U is orthogonal

        # testing on a single block, with blockSize = n for now - ALEC
        A = []
        blockSize = 8

        # create the nxn array
        for i in range(blockSize):
            A.append(numpy.arange(blockSize*i, blockSize*(i+1)))

        A = numpy.array(A)

        # compute the SVD
        res = self.computeSVD(A)

        print(res[0])
        print()
        print(res[1])
        print()
        print(res[2])

        # reconstruction
        '''
        B = U.dot(Sigma.dot(VT))

        print()
        print(B)
        '''

        return None

    def decode(self):
        """Decode message from image."""
        # TODO: Implement
        return None

    def run(self):
        """Run Steganography class."""
        if self.method == "embed":
            print("RUNNING steganographer with METHOD embed")
            self.embed()
        else:
            print("RUNNING steganographer with METHOD decode")
            self.decode()

        #self.output()
