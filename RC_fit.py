import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class RC_fit:
    def __init__(self, data):
        '''
        Initialize the class with data

        Parameters:
            data: pandas dataframe with columns 'Time', 'Resistance', 'Error'
        '''
        self.data = data
        self.x = data[data['Cycle'] != 'Pre-Cycling']['Time']
        self.y = data[data['Cycle'] != 'Pre-Cycling']['Resistance']
        self.baseline_x = data[data['Cycle'] == 'Pre-Cycling']['Time']
        self.baseline_y = data[data['Cycle'] == 'Pre-Cycling']['Resistance']

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

    def baseline(self):
        '''
        Subtract the baseline from the data
        '''
        popt, _ = curve_fit(self.baseline_model, self.baseline_x, self.baseline_y)
        self.baseline = self.baseline_model(self.x, *popt)

        # Subtract the baseline from the full dataset
        self.data['Resistance_corrected'] = self.data['Resistance'] - self.baseline_model(self.data['Time'], *popt)

        # Separate the "on" and "off" cycles here
        self.on = self.data[self.data['Cycle'].str.contains('On', case=False)].copy()
        self.off = self.data[self.data['Cycle'].str.contains('Off', case=False)].copy()

    def fit(self):
        '''
        Fit the model to the data
        '''
        # Shift time so that t = 0 at the start
        self.on.loc[:, 'Time_shifted'] = self.on['Time'] - self.on['Time'].min()
        self.off.loc[:, 'Time_shifted'] = self.off['Time'] - self.off['Time'].min()

        # Subtract the baseline from the "on" and "off" cycles using .loc to avoid warnings
        self.on.loc[:, 'Resistance_corrected'] = self.on['Resistance'] - self.baseline_model(self.on['Time'], *curve_fit(self.baseline_model, self.baseline_x, self.baseline_y)[0])
        self.off.loc[:, 'Resistance_corrected'] = self.off['Resistance'] - self.baseline_model(self.off['Time'], *curve_fit(self.baseline_model, self.baseline_x, self.baseline_y)[0])

        # Provide better initial guesses based on the observed data
        R0_on_guess = self.on['Resistance_corrected'].max()  # Initial resistance guess for "on"
        Rf_on_guess = self.on['Resistance_corrected'].min()  # Final resistance guess for "on"
        C_on_guess = 10  # Time constant guess for on cycle (adjust this if necessary)

        # Fit the "on" cycle data with shifted time (fitting both R0 and Rf)
        popt_on, _ = curve_fit(self.RC_on_model, self.on['Time_shifted'], self.on['Resistance_corrected'], p0=[R0_on_guess, Rf_on_guess, C_on_guess])
        self.popt_on = popt_on

        # Use the last value from the "on" cycle as the initial guess for Rf in the "off" cycle
        Rf_off_guess = self.on['Resistance_corrected'].iloc[-1]  # Last value in the on cycle
        R0_off_guess = self.off['Resistance_corrected'].max()  # Initial guess for R0 in the off cycle
        C_off_guess = 10  # Time constant guess for off cycle

        # Fit the "off" cycle data with shifted time and use Rf_off_guess
        popt_off, _ = curve_fit(self.RC_off_model, self.off['Time_shifted'], self.off['Resistance_corrected'], p0=[R0_off_guess, Rf_off_guess, C_off_guess])
        self.popt_off = popt_off


    def plot(self):
        '''
        Plot the fitted model
        '''
        # Plot the "on" cycle data with the fitted model using shifted time
        plt.plot(self.on['Time_shifted'], self.on['Resistance_corrected'], 'bo', label='On Data (Corrected)')
        plt.plot(self.on['Time_shifted'], self.RC_on_model(self.on['Time_shifted'], *self.popt_on), 'b-', label='On Fit')

        # Plot the "off" cycle data with the fitted model using shifted time
        plt.plot(self.off['Time_shifted'], self.off['Resistance_corrected'], 'ro', label='Off Data (Corrected)')
        plt.plot(self.off['Time_shifted'], self.RC_off_model(self.off['Time_shifted'], *self.popt_off), 'r-', label='Off Fit')

        plt.xlabel('Time')
        plt.ylabel('Resistance (Corrected)')
        plt.legend()
        plt.show()

    def get_parameters(self):
        '''
        Return the fitted parameters and time constants (tau) for both "on" and "off" cycles
        '''
        R0_on, Rf_on, tau_on = self.popt_on
        R0_off, Rf_off, tau_off = self.popt_off
        
        return {
            "on": {
                "Delta_R": int(R0_on - Rf_on),
                "tau": int(tau_on)
            },
            "off": {
                "Delta_R": int(R0_off - Rf_off),
                "tau": int(tau_off)
            }
        }


if __name__ == '__main__':
    data = pd.read_csv('20241007_PN1_CuOxSnOx_EtOH_standard_25ppm.csv')
    rc = RC_fit(data)
    rc.baseline()
    rc.fit()
    rc.plot()
    print(rc.get_parameters())
