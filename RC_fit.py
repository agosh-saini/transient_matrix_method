######################################################
# Author: Agosh Saini
# Date: 2024-10-25
# Contact: contact@agoshsaini.com
######################################################
# Description: This script fits the RC model to the data
######################################################


######### IMPORTS #########
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

######### CLASS DEFINITION #########
class RC_fit:
    def __init__(self, data=None, on=None, off=None, baseline=None, timestep=None):
        '''
        Initialize the class with either data or predefined on, off, and baseline cycles.
        If no Time column is present, the time array is generated based on the timestep.

        Parameters:
            data: pandas dataframe with columns 'Time', 'Resistance', 'Error' (optional)
            on: numpy array for 'on' cycle resistance data (optional)
            off: numpy array for 'off' cycle resistance data (optional)
            baseline: numpy array for baseline resistance data (optional)
            timestep: average timestep between data points (optional, required if no Time column is present)
        '''
        self.data = data
        self.on = on
        self.off = off
        self.baseline_data = baseline
        self.timestep = timestep

        if self.data is not None:
            self.x = data[data['Cycle'] != 'Pre-Cycling']['Time']
            self.y = data[data['Cycle'] != 'Pre-Cycling']['Resistance']
            self.baseline_x = data[data['Cycle'] == 'Pre-Cycling']['Time']
            self.baseline_y = data[data['Cycle'] == 'Pre-Cycling']['Resistance']
            self.preprocess_data()
        elif self.on is None or self.off is None or self.baseline_data is None:
            raise ValueError("If no data is provided, 'on', 'off', and 'baseline' must be provided.")
        else:
            # Handle missing 'Time' data by generating it based on the timestep
            num_on_points = len(self.on)
            num_off_points = len(self.off)
            num_baseline_points = len(self.baseline_data)

            if self.timestep is None:
                raise ValueError("Timestep must be provided if the 'Time' column is missing.")
                
            # Generate Time arrays based on the number of data points and timestep
            self.on_time = np.arange(num_on_points) * self.timestep
            self.off_time = np.arange(num_off_points) * self.timestep
            self.baseline_time = np.arange(num_baseline_points) * self.timestep

    def preprocess_data(self):
        '''
        Preprocess the data if data is provided as a DataFrame.
        This includes setting up on, off, and baseline cycles.
        '''
        # Subtract the baseline from the data
        self.baseline()
        self.on = self.data[self.data['Cycle'].str.contains('On', case=False)].copy()
        self.off = self.data[self.data['Cycle'].str.contains('Off', case=False)].copy()
    
    def extract_baseline(self, seconds=100):
        '''
        Calculate the baseline resistance using the last 'seconds' of data
        '''
        self.baseline_data = self.y[-seconds:]
        self.baseline_time = self.x[-seconds:]
    
    def baseline_model(self, x, m, c):
        '''
        Model function for the baseline
        Parameters:
            x: Time
            y: Resistance
            m: Slope
            c: Intercept
        Returns:
            y: Resistance
        '''
        return m * x + c
    
    def fit_baseline(self):
        '''
        Fit the baseline resistance using the last 'seconds' of data
        '''
        popt, _ = curve_fit(self.baseline_model, self.baseline_time, self.baseline_data)
        self.baseline = self.baseline_model(self.baseline_time, *popt)

        self.on_resistance_corrected = self.on - self.baseline_model(self.on_time, *popt)
        self.off_resistance_corrected = self.off - self.baseline_model(self.off_time, *popt)


    def RC_on_model(self, t, R0, Rf, C):
        '''
        Model function for the RC on model
        Parameters:
            t: Time
            R0: Initial resistance
            Rf: Final resistance
            C: Time constant
        Returns:
            y: Resistance
        '''
        return Rf + (R0 - Rf) * np.exp(-t / C)

    def RC_off_model(self, t, R0, Rf, C):
        '''
        Model function for the RC off model
        Parameters:
            t: Time
            R0: Initial resistance
            Rf: Final resistance
            C: Time constant
        Returns:
            y: Resistance
        '''
        return Rf + (R0 - Rf) * np.exp(-t / C)
    

    def compute_r_squared(self, y_true, y_pred):
        '''
        Compute the R-squared value for the model
        '''
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared


    def fit(self, r_squared_threshold=0.9):
        '''
        Fit the model to the data if R^2 exceeds a threshold.
        '''
        # Shift time so that t = 0 at the start
        self.on_time_shifted = self.on_time - self.on_time.min()
        self.off_time_shifted = self.off_time - self.off_time.min()

        # Subtract the baseline from the "on" and "off" cycles
        self.fit_baseline()

        # Initial guesses based on the observed data
        R0_on_guess = self.on_resistance_corrected.max()  # Initial resistance guess for "on"
        Rf_on_guess = self.on_resistance_corrected.min()  # Final resistance guess for "on"
        C_on_guess = 10  # Time constant guess for on cycle (adjust if necessary)

        # Fit the "on" cycle data with shifted time
        popt_on, _ = curve_fit(self.RC_on_model, self.on_time_shifted, self.on_resistance_corrected, p0=[R0_on_guess, Rf_on_guess, C_on_guess])
        self.popt_on = popt_on

        # Calculate predicted values and R^2 for "on" cycle
        on_predicted = self.RC_on_model(self.on_time_shifted, *popt_on)
        r_squared_on = self.compute_r_squared(self.on_resistance_corrected, on_predicted)

        # Check R^2 for "on" cycle
        if r_squared_on < r_squared_threshold:
            self.tau_on = 0
            self.delta_r_on = 0
        else:
            self.tau_on = popt_on[2]
            self.delta_r_on = popt_on[0] - popt_on[1]

        # Use the last value from the "on" cycle as the initial guess for Rf in the "off" cycle
        Rf_off_guess = self.on_resistance_corrected[-1]  # Last value in the on cycle
        R0_off_guess = self.off_resistance_corrected.max()  # Initial guess for R0 in the off cycle
        C_off_guess = 10  # Time constant guess for off cycle

        # Fit the "off" cycle data with shifted time
        popt_off, _ = curve_fit(self.RC_off_model, self.off_time_shifted, self.off_resistance_corrected, p0=[R0_off_guess, Rf_off_guess, C_off_guess])
        self.popt_off = popt_off

        # Calculate predicted values and R^2 for "off" cycle
        off_predicted = self.RC_off_model(self.off_time_shifted, *popt_off)
        r_squared_off = self.compute_r_squared(self.off_resistance_corrected, off_predicted)

        # Check R^2 for "off" cycle
        if r_squared_off < r_squared_threshold:
            self.tau_off = 0
            self.delta_r_off = 0
        else:
            self.tau_off = popt_off[2]
            self.delta_r_off = popt_off[0] - popt_off[1]


    def plot(self, show_plot=False, save_plot=False, file_name=None, graph_folder=None):
        '''
        Plot the fitted model
        '''
        # Plot the "on" cycle data with the fitted model using shifted time
        plt.plot(self.on_time_shifted, self.on_resistance_corrected, 'bo', label='On Data (Corrected)')
        plt.plot(self.on_time_shifted, self.RC_on_model(self.on_time_shifted, *self.popt_on), 'b-', label='On Fit')

        # Plot the "off" cycle data with the fitted model using shifted time
        plt.plot(self.off_time_shifted, self.off_resistance_corrected, 'ro', label='Off Data (Corrected)')
        plt.plot(self.off_time_shifted, self.RC_off_model(self.off_time_shifted, *self.popt_off), 'r-', label='Off Fit')

        plt.xlabel('Time')
        plt.ylabel('Resistance (Corrected)')
        plt.legend()

        if show_plot:
            plt.show()

        if save_plot:

            if file_name is None:
                file_name = f'{time.time()}_RC_plot'
            
            if graph_folder is None:
                graph_folder = 'graph_folder'
            
            if not os.path.exists(graph_folder):
                os.makedirs(graph_folder)

            file_name = os.path.basename(file_name)

            plt.savefig(f'{graph_folder}/{file_name}.png')
            plt.close()

        
    def get_parameters(self):
        '''
        Return the fitted parameters and time constants (tau) for both "on" and "off" cycles
        '''
        _, _, tau_on = self.popt_on
        _, _, tau_off = self.popt_off

        R_on = self.on[-1]
        R_off = self.off[-1]

        R0_on = self.on[0]
        R0_off = self.off[0]
        
        return {
            "on": {
                "Delta_R": R0_on - R_on,
                "tau": tau_on,
                "R0": R0_on
            },
            "off": {
                "Delta_R": R0_off - R_off,
                "tau": tau_off,
                "R0": R0_off
            }
        }
    

######### MAIN FUNCTION #########
if __name__ == '__main__':

    def generate_data(x,dir=1, seed=42):
        '''
        Generate example data for testing
        '''

        random = np.random.RandomState(seed)
        y = 100 + 50 * np.exp(-dir*x / 10) + random.normal(0, 5, x.size)
        return y


    # Example: Directly passing pre-processed 'on', 'off', and 'baseline' cycles as NumPy arrays
    baseline_cycle = np.linspace(100, 200, 50)  # Baseline resistance
    on_cycle = generate_data(np.linspace(0, 100, 100))  # Resistance data for "on" cycle
    off_cycle = generate_data(np.linspace(0, 100, 100), -1)  # Resistance data for "off" cycle

    # Assuming an average timestep of 0.5 seconds
    timestep = 0.5

    rc = RC_fit(on=on_cycle, off=off_cycle, baseline=baseline_cycle, timestep=timestep)
    rc.fit()
    rc.plot(save_plot=True, show_plot=True, file_name='RC_plot', graph_folder='graph_folder') 
    print(rc.get_parameters())

