"""Steganographer Tool."""
import os
import imageio
import click
from svd_steg.steganographer import Steganographer


# TODO: Fix help messages in click
@click.command()
@click.option('--embed', '-e', default=False, is_flag=True, help='')
@click.option('--decode', '-d', default=False, is_flag=True, help='')
@click.argument('message_file')
@click.argument('image_file')
def main(embed, decode, image_file, message_file):
    """Run Steganographer tool."""
    method = ""
    if embed:
        print("METHOD embed")
        method = "embed"
    elif decode:
        method = "decode"
        print("METHOD decode")

    input_dir = "input"

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
    print(method)
    stego = Steganographer(method, image_in,
                           image_file, message_in, message_file)
    stego.run()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
