"""Steganographer Class"""
import os
import numpy
import imageio
from steganographer.helper import progressBar

class Steganographer:

	def __init__(self,  method, image_in, image_file, message_in="", message_file=""):
		self.message = message_in
		self.message_file = message_file
		self.image = image_in.astype(numpy.int32)
		self.image_file = image_file
		self.method = method
		self.embedded_image = numpy.zeros(self.image.shape)

	def output_embedded_image(self):
	    file_split = self.image_file.split('.')
	    output_filename = file_split[0] + '_steg.' + file_split[1]

	    imageio.imwrite("output/" + output_filename, self.embedded_image)

	def output_decoded_text(self):
		file_split = self.image_file.split('.')
	    output_filename = file_split[0] + '_text.txt'

		file = open("ouput/" + output_filename, 'w+')
		file.write(self.message)
		file.close()

	def output(self):
		if not os.path.exists(output_dir):
        	os.makedirs(output_dir)
        if self.method = "embed":
        	self.output_embedded_image()
        else:
        	self.output_decoded_text()

    def embed(self):
    	# TODO: Implement
    	return None

    def decode(self):
    	# TODO: Implement
    	return None

	def run(self):
		if self.method == "embed":
			self.embed()
		else:
			self.decode()
	
		self.output()
