# SVD_Steg
Steganography tool using Singular Value Decomposition

To set up a python environment for this project please run
````
python3 -m venv env
````
### Important:
Also, remember to activate your virtual environment EVERYTIME you are using/working on the project:
````
source env/bin/activate
````
Also make sure to install everything from setup.py
````
pip install -e .
````
You may get a small error message for imageio installation but ignore it.

Make sure to make the script executable
````
chmod +x bin/svdsteg
````
## Note
If the script does not run in the terminal, it may be because it thinks you have windows line endings instead of unix. This is fairly easy to change in the IDE/text editor of your choice.

# Commiting
Make sure to check the style of all your code before commiting:
````
./bin/svdsteg style
````

# Running the tool
The SVD Steg is run from the svdsteg script which can be used as follows
````
./bin/svdsteg COMMAND FILE1 FILE2
````
In the above, COMMAND is a non-optional argument that must either be embed if you are trying to embed a message, decrypt to decrypt a message, or style to check the style of the codebase.
FILE1 is non optional when the COMMAND is embed or decrypt. For encrypt FILE1 should be a .txt file holding the message desired to be embedded For decrypt FILE1 should be an image file holding a hidden message.
FILE2 is non optional when the COMMAND is embed and should be the image file you wish to hide the message in.
