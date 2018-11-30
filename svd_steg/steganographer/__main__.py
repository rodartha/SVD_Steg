"""Steganographer Tool"""
import os
import click
from steganographer.steganographer import Steganographer


# TODO: Fix help messages in click
@click.command()
@click.option('--embed', '-e', default=False, is_flag=True, help='')
@click.option('--decode', '-d', default=False, is_flag=True, help='')
@click.argument('message_file')
@click.argument('image_file')
def main(embed, decode, message_file, image_file):
    """Run Steganographer tool."""
    method = ""
    if embed:
        method = "encode"
    elif decode:
        method = "decode"

    input_dir = "input"
    output_dir = "output"

    # Load Image
    if not os.path.exists(input_dir):
        print("Error: Input Directory does not exist")
        exit(1)
    if not os.path.isfile(input_dir + '/' + image_file):
        print("Error: Image file "
              + image_file + " does not exist in input folder.")
        exit(1)
    image_in = imageio.imread(input_dir + '/' + image_file)

    # Load Message
    if not os.path.isfile(input_dir + '/' + message_file):
    	print("Error: Message file " 
    		  + message_file + " does not exit in input folder.")
    file = open(input_dir + '/' + message_file, 'r')
    message_in = file.read()

    # Run Steganography Tool
    stego = Steganographer(method, image_in, image_file, message_in, message_file)
    stego.run()

if __name__ == '__main__':
	# pylint: disable=no-value-for-parameter
	main()
