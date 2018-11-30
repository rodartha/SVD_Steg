# SVD_Steg
Steganography tool using Singular Value Decomposition

To set up a python environment for this project please run
````
python3 -m venv env
````
Also, remember to activate your virtual environment whenever using/working on the project:
````
source env/bin/activate
````
Also make sure to install everything from setup.py
````
pip install -e .
````
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
