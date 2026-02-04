import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


from GPassets.preprocessing import PreProcessor
from GPassets.model import GPModel
import numpy as np
import fpsample
from GPassets.plotting import plot_errors

EXAMPLE = "Robot Example"

# Load Data 
if EXAMPLE.casefold() == 'Robot Example'.casefold():
    X_full = np.load(f'{PROJECT_ROOT}\\GP Regression\\GPassets\\data\\X_i_robot_AGM.npy')
    Lambda_full = np.load(f'{PROJECT_ROOT}\\GP Regression\\GPassets\\data\\Y_i_robot_AGM.npy')
elif EXAMPLE.casefold() == 'Sensor Example'.casefold():
    X_full = np.load(f'{PROJECT_ROOT}\\GP Regression\\GPassets\\data\\X_i_sensor_AGM.npy')
    Lambda_full = np.load(f'{PROJECT_ROOT}\\GP Regression\\GPassets\\data\\Y_i_sensor_AGM.npy')

Nsamp = len(X_full)

# Subsample data
k = 5000
sample_idx = fpsample.fps_sampling(X_full, k, 50)

X = X_full[sample_idx]
Lambda = Lambda_full[sample_idx]

# Create pre-processing object
prep_obj = PreProcessor()

# Data pre-processing step
x_train, x_test, lambda_train, lambda_test = prep_obj.prep_data(X, Lambda, debug=False)

# Save pre-processing assets
prep_obj.save(f'{PROJECT_ROOT}\\GP Regression\\GPassets\\prep_assets_{EXAMPLE}.pkl')

train = False
test = True

kernel_type =  "Matern12" # "Matern12" OR "ReLU"
version = 3 # 1 OR 2 OR 3 OR 4(1- default initialization, 2,3 - special initializations)
device = 'cuda'

gp_config = {'kernel' : kernel_type,
             'version': version,
             'depth' : 1,
            }

# Create GPModel object with GP config
gp = GPModel(gp_config, prep_obj, device)

model_savepath = f'{PROJECT_ROOT}\\GP Regression\\GPassets\\models\\{EXAMPLE}\\Model_{kernel_type}_{version}.pth' 

# ------------------------------- GP Training ----------------------------------------
if train:

    train_report_path = f'{PROJECT_ROOT}\\GP Regression\\Results\\reports\\{EXAMPLE}\\Train_report_{kernel_type}_{version}.txt'    

    # Build GP model
    gp.build_gp_model(x_train=x_train, y_train=lambda_train, device='cuda')

    # Train model
    gp.train_model(savepath=train_report_path)

    # Save GP model 
    gp.save_model(model_savepath)

# ------------------------------- GP Testing ----------------------------------------
if test:

    test_report_path  = f'{PROJECT_ROOT}\\GP Regression\\Results\\reports\\{EXAMPLE}\\Test_report_{kernel_type}_{version}.txt'    

    # Load GP model (sets to eval mode internally)
    gp.load_model(model_savepath)

    # Model inference analysis
    training_error, testing_error = gp.analyse_model(x_train=x_train, y_train=lambda_train, x_test_np=x_test, y_test=lambda_test, path = test_report_path, generate_report=True)  

    # Plot training and testing errors
    # plot_filename = f'{PROJECT_ROOT}\\GP Regression\\plots\\{EXAMPLE}\\err_{kernel_type}_{version}.pdf'
    # plot_errors(training_error['e_sample'], testing_error['e_sample'], f'{kernel_type} kernel', plot_filename, annotate=True)




    