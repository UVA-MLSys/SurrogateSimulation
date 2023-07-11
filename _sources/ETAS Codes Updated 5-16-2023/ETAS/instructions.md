## How to get Started:

##### Step 1: Download Anaconda:
- Visit the Anaconda website: https://www.anaconda.com/products/individual.
- Download the Anaconda distribution appropriate for your operating system (Windows, macOS, or Linux).
    - Choose the Python 3.x version

&nbsp;

##### Step 2: Open Anaconda Navigator:
- Once Anaconda is successfully installed, open the Anaconda Navigator application which acts as a GUI

&nbsp;

##### Step 3: Create a New Environment:
- In the Anaconda Navigator window, click on the "Environments" tab on the left sidebar.
- Click the "Create" button at the bottom left corner of the sidebar to create a new environment.
- Enter a name for your new environment (e.g., "myenv") in the "Name" field.
- Select the Python version you want to use for the environment (e.g., Python 3.7).
- Install the packages from the "Anaconda" Click the "Create" button to create the new environment.
- Click on the Search Packages search bar in the right hand corner of the window and search for a packag “cartopy” and install it

&nbsp;

##### Step 4: Activate the Environment:
- Once the environment is created, go back to the "Home" tab in Anaconda Navigator.
- Click on the dropdown menu next to the "Applications on" text.
- Select the environment you just created (e.g., "myenv") from the list.

&nbsp;

##### Step 5: Launch IDE or Jupyter Notebook:
- In the Home Tab, launch VSCode, PyCharm or your IDE of choice or a Jupyter Notebook file
    - This will open the chosen IDE within your new Anaconda environment, ready for use.  

&nbsp;

##### Step 6: Installing Packages:
- Inside your IDE, navigate to the directory of the launched project using the cd command (if the directory is not already at the root directory of the project)
- Install all required pip packages
    - For this code the pip packages required are numpy, matplotlib
    - Can be installed using the command: pip install name (where name is the name of the package)
- Install all required conda packages
    - For this code the pip packages required are cartopy
    - Can be installed using the command: conda install name (where name is the name of the package)  

&nbsp;

##### Step 7: Running the Code:
- Run the ETAS_ChatGPT_V2.12.py file by using the run button from your IDE or by using the command python3 ETAS_ChatGPT_V2.12.py