# ECE471_duckhunt_spring2022
Duck hunt video game computer vision class project solution

*also note it performs best on level 30 because this is where most of the training data came from*

***
files that were modified/needed to run the solution :

solution.py -> solution added

duck_hunt_main.py -> added few lines to have coordinates be a list of up to 10 coordinates

model_ws_64.mat -> model needed for the binary classification in solution.py
***


To start the virtual environment:

.\myvenvs\Scripts\activate

To run the code: (level 30 for example)

C:\>python C:\myvenvs\Project\ece471_536-S2022-main\ece471_536-S2022-main\duck-hunt\duck_hunt_main.py -m absolute -l 30

***
Note the model should be in the same directory as solution.py as it is loaded from this line in solution.py:

model_ws_64 = scipy.io.loadmat('C:/myvenvs/Project/ece471_536-S2022-main/ece471_536-S2022-main/duck-hunt/model_ws_64.mat')
***


This is not needed but this was the command used to train the model:
[D_64_train, D_64_test] = set_up_dataset_64();
[ws,fs,k] = grad_desc_mod('duckhunt_f_wdbc','duckhunt_g_wdbc',zeros(1765,1),250,T_64_train,0.075);

solution.py and duck_hunt_main.py were only files modified that are needed to run the solution (and the MATLAB file with the model weights is needed too)
