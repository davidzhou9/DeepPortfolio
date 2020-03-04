"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""
import math

import os
import numpy as np
import tensorflow as tf
from config import get_config
from equation import get_equation
from solver import FeedForwardModel
import warnings
from scipy.stats import multivariate_normal as normal
warnings.simplefilter("ignore")

TF_DTYPE = tf.float64
FLAGS = tf.compat.v1.flags.FLAGS
#FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('problem_name', 'HJB',
                           """The name of partial differential equation.""")
tf.app.flags.DEFINE_integer('num_run', 1,
                            """The number of experiments to repeatedly run for the same problem.""")
tf.app.flags.DEFINE_string('log_dir1', './logs',
                           """Directory where to write event logs and output array.""")


def main():

    problem_name = FLAGS.problem_name # get probelm name
    config = get_config(problem_name)
    
    if not os.path.exists(FLAGS.log_dir1): # check to see if this log directory already exists
        os.mkdir(FLAGS.log_dir1)
    path_prefix = os.path.join(FLAGS.log_dir1, problem_name) # create the name of the path file
    
    total_Time = config.total_time
    num_time_interval = config.num_time_interval
    dt = total_Time / num_time_interval
    
    print("----- BEFORE ITERS ----- ")
    print("total_Time: ", total_Time)
    print("num_time_interval: ", num_time_interval)
    print("dt: ", dt)
    
    ### CALCULATING PORTFOLIOS
    bsde = get_equation(problem_name, config.dim, total_Time, num_time_interval)
    wealth = bsde.wealth()
    gamma = bsde.gamma()
    mu = bsde.mu()
    r = bsde.interest_Rate()
    alpha_revert = bsde.alpha_Revert()
    delta_revert = bsde.delta_Revert()
    rho_1 = bsde.rho1()
    rho_2 = bsde.rho2()
    rho_12 = bsde.rho12()
    nu_f = bsde.nu_f()
    nu_s = bsde.nu_s()
    mf = bsde.muF()
    ms = bsde.muS()
        
    print("wealth: ", wealth)
    print("gamma: ", gamma)
    print("mu: ", mu)
    print("r: ", r)
    print("alpha_revert: ", alpha_revert)
    print("delta_revert: ", delta_revert)
    print("rho1: ", rho_1)
    print("rho2: ", rho_2)
    print("rho12: ", rho_12)
    print("nu_f: ", nu_f)
    print("nu_s: ", nu_s)
    print("mf: ", mf)
    print("ms: ", ms)
    
    num_Samples = 2
    dw_Factors, process_Factors, stockPrice = simStockMultiscale(num_Samples, mu, r, alpha_revert, delta_revert, rho_1, rho_2, rho_12, nu_f, nu_s, dt, num_time_interval, mf, ms)
    
    print("all dw_Factors: ", dw_Factors)    
    print("all process_Factors: ", process_Factors)
    print("all stockPrices: ", stockPrice)    

    merton_Sample_Payoffs = np.array([])
    merton_Portfolio_Value_Over_Time = np.array([])
    merton_Strats_Over_Time = np.array([])
    
    NN_Sample_Payoffs = np.array([])
    NN_Portfolio_Value_Over_Time = np.array([])
    NN_Strats_Over_Time = np.array([])
    
    print("------------------ PORTFOLIO STAGE ----------------------")
    nn_Position_In_Stock = 0 # gives number of stocks purchased
    nn_Wealth_In_Bonds = 0 # gives amount of money in risk free
    nn_Portfolio_Value = wealth
                
    merton_Position_In_Stock = 0 # gives number of stocks purchased
    merton_Wealth_In_Bonds = 0
    merton_Portfolio_Value = wealth
    
    # fix the first sampel sample
    dw_Factor_Sample = dw_Factors[0, :, :]
    process_Factors_Sample = process_Factors[0, :, :]
    
    print("dw_Factor_Sample: ", dw_Factor_Sample)
    print("process_Factors_Sample: ", process_Factors_Sample)
    print("\n")
    for time_Index in range(num_time_interval):
        tf.reset_default_graph()
        
        print("------------------ Time_Index: ", time_Index, " ------------------------- \n")
        with tf.Session() as sess:
            remaining_Time = total_Time - time_Index * dt 
            remaining_Intervals = num_time_interval - time_Index
            
            print("remaining Time: ", remaining_Time)
            print("remaining Intervals: ", remaining_Intervals)
            #logging.info('Begin to solve %s with run %d' % (problem_name, idx_run))
            #config.setTotalTime(remaining_Time)
            print("config.total_time before change: ", config.total_time)
            print("config.num_time_interval before change: ", config.num_time_interval)
            
            config.total_time = remaining_Time
            config.num_time_interval = remaining_Intervals
            
            print("config.total_time after change: ", config.total_time)
            print("config.num_time_interval after change: ", config.num_time_interval)
            #config.setNumTimeIntervals(remaining_Intervals)
            
            
            ### change the factors to update to current time
            curr_Factors = process_Factors[0, :, time_Index] # get x (y, z if exists) at first time step
            curr_Y_Factor = process_Factors[0, 0, time_Index]
            curr_Z_Factor = process_Factors[0, 1, time_Index]
            curr_Stock_Price = stockPrice[0, 0, time_Index]
            print("curr_Factors: ", curr_Factors)
            print("curr_Stock_Price: ", curr_Stock_Price)
            print("curr_Y_Factor: ", curr_Y_Factor)
            print("curr_Z_Factor: ", curr_Z_Factor)
            
            bsde = get_equation(problem_name, config.dim, remaining_Time, remaining_Intervals)
            bsde.setY(curr_Y_Factor)
            bsde.setZ(curr_Z_Factor)
            
            print("bsde Y Factor: ", bsde.getY())
            print("bsde Z Factor: ", bsde.getZ())
            
            ###### IMPORTANT ######
            model = FeedForwardModel(config, bsde, sess, problem_name) # create the feed forward model
            #if bsde.y_init:
            #    logging.info('Y0_true: %.4e' % bsde.y_init)
            model.build() # bulid the model
            training_history = model.train() # trin the model
            #if bsde.y_init:
            #    logging.info('relative error of Y0: %s',
            #                 '{:.2%}'.format(
            #                     abs(bsde.y_init - training_history[-1, 2])/bsde.y_init))
            # save training history
            
            #training_history = np.append(training_history, bsde._total_time)
            np.savetxt('{}_training_history_{}.csv'.format(path_prefix, time_Index),
                       training_history,
                       fmt=['%d', '%.5e', '%.5e', '%d'],
                       delimiter=",",
                       header="step,loss_function,target_value,elapsed_time",
                       comments='')
            
            print("dw_Factor_Sampe[:, time_Index:]: ", dw_Factor_Sample[:, time_Index:])
            print("process_Factors_Sample[:, time_Index]: ", process_Factors_Sample[:, time_Index:])
            
            init_Out, output_Z_Vals = model.simPortfolio()
            
            current_Sample = output_Z_Vals[:, 0, :]
            print("init_Out: ", init_Out)
            print("output_Z_Vals: ", output_Z_Vals)
            print("current_Sample: ", current_Sample)
            print("current_Sample[0, :]: ", current_Sample[0, :])
               
            mat_Sqrt = model.calculate_Diffusion_Mat(curr_Factors)
            grads = np.linalg.solve(np.transpose(mat_Sqrt), current_Sample[0, :])
            print("mat_Sqrt: ", mat_Sqrt)
            print("grads: ", grads)
                
            g_y = grads[0]
            g_z = grads[1]
            
            V_x = wealth**(gamma - 1) * math.exp(-init_Out)
            V_xx = (gamma - 1) * wealth**(gamma - 2) * math.exp(-init_Out)
                
            V_xy = -g_y * wealth**(gamma - 1) * math.exp(-init_Out)
            V_xz = -g_z * wealth**(gamma - 1) * math.exp(-init_Out)
            
            nn_pi_Strategy = -((mu - r) * V_x + math.sqrt(2 * alpha_revert) * nu_f * rho_1 * math.exp(curr_Y_Factor + curr_Z_Factor) * V_xy + math.sqrt(2 * delta_revert) * nu_s * rho_2 * math.exp(curr_Y_Factor + curr_Z_Factor) * V_xz) / (math.exp(2 * (curr_Y_Factor + curr_Z_Factor)) * wealth * V_xx)
            merton_Strategy = (mu - r) / (math.exp(2 * (curr_Y_Factor + curr_Z_Factor)) * (1 - gamma))
                
            print("nn_pi_Strategy: ", nn_pi_Strategy)
            print("merton_Strategy: ", merton_Strategy)
                
            # if initial time create the portfolios
            if time_Index == 0:
                nn_Position_In_Stock = (nn_pi_Strategy * wealth) / curr_Stock_Price # gives number of stocks purchased
                nn_Wealth_In_Bonds = wealth - nn_Position_In_Stock * curr_Stock_Price # gives amount of money in risk free
                nn_Portfolio_Value = wealth
                
                merton_Position_In_Stock = (merton_Strategy * wealth) / curr_Stock_Price # gives number of stocks purchased
                merton_Wealth_In_Bonds = wealth - merton_Position_In_Stock * curr_Stock_Price
                merton_Portfolio_Value = wealth
            else: # rebalance
                # update NN Portfolio value
                nn_Portfolio_Value = nn_Wealth_In_Bonds * math.exp(r * dt) + curr_Stock_Price * nn_Position_In_Stock
                print("nn portfolio after change: ", nn_Portfolio_Value)
                    
                # update merton Portfolio value
                merton_Portfolio_Value = merton_Wealth_In_Bonds * math.exp(r * dt) + curr_Stock_Price * merton_Position_In_Stock
                print("merton portfolio after change: ", merton_Portfolio_Value)
                
                # rebalance portfolios
                nn_Position_In_Stock = (nn_pi_Strategy * nn_Portfolio_Value) / curr_Stock_Price # gives number of stocks purchased
                nn_Wealth_In_Bonds = nn_Portfolio_Value - nn_Position_In_Stock * curr_Stock_Price # gives amount of money in risk free
                
                merton_Position_In_Stock = (merton_Strategy * merton_Portfolio_Value) / curr_Stock_Price # gives number of stocks purchased
                merton_Wealth_In_Bonds = merton_Portfolio_Value - merton_Position_In_Stock * curr_Stock_Price
            
            merton_Portfolio_Value_Over_Time = np.append(merton_Portfolio_Value_Over_Time, merton_Portfolio_Value)
            NN_Portfolio_Value_Over_Time = np.append(NN_Portfolio_Value_Over_Time, nn_Portfolio_Value)
            
            merton_Strats_Over_Time = np.append(merton_Strats_Over_Time, merton_Position_In_Stock)
            NN_Strats_Over_Time = np.append(NN_Strats_Over_Time, nn_Position_In_Stock)
            
            print("nn_Position_In_Stock: ", nn_Position_In_Stock)
            print("nn_Wealth_In_Bonds: ", nn_Wealth_In_Bonds)
            
            print("merton_Position_In_Stock: ", merton_Position_In_Stock)
            print("merton_Wealth_In_Bonds: ", merton_Wealth_In_Bonds)   
            print("\n\n")
                
    print("\n ----------- FINAL REBALANCE ------------------ \n")
    curr_Stock_Price = stockPrice[0, 0, -1]
    nn_Portfolio_Value = nn_Wealth_In_Bonds * math.exp(r * dt) + curr_Stock_Price * nn_Position_In_Stock
    print("nn portfolio after change: ", nn_Portfolio_Value)
                    
    # update merton Portfolio value
    merton_Portfolio_Value = merton_Wealth_In_Bonds * math.exp(r * dt) + curr_Stock_Price * merton_Position_In_Stock
    print("merton portfolio after change: ", merton_Portfolio_Value)
    
    # add final rebalance
    merton_Portfolio_Value_Over_Time = np.append(merton_Portfolio_Value_Over_Time, merton_Portfolio_Value)
    NN_Portfolio_Value_Over_Time = np.append(NN_Portfolio_Value_Over_Time, nn_Portfolio_Value)
    
    merton_Strats_Over_Time = np.append(merton_Strats_Over_Time, merton_Position_In_Stock)
    NN_Strats_Over_Time = np.append(NN_Strats_Over_Time, nn_Position_In_Stock)
    
    nn_Payoff = (nn_Portfolio_Value)**gamma / gamma
    merton_Payoff = (merton_Portfolio_Value)**gamma / gamma
    
    print("nn_Payoff: ", nn_Payoff)
    print("merton_Payoff: ", merton_Payoff)
            
    NN_Sample_Payoffs = np.append(NN_Sample_Payoffs, nn_Payoff)
    merton_Sample_Payoffs = np.append(merton_Sample_Payoffs, merton_Payoff)
    
    np.savetxt('{}_NNPandL.csv'.format(path_prefix),
                           NN_Sample_Payoffs,
                           fmt=['%.5e'],
                           delimiter=",",
                           header="Samples",
                           comments='')
    
    np.savetxt('{}_MertonPandL.csv'.format(path_prefix),
                           merton_Sample_Payoffs,
                           fmt=['%.5e'],
                           delimiter=",",
                           header="Samples",
                           comments='')

    np.savetxt('{}_NNValueOverTime.csv'.format(path_prefix),
                           NN_Portfolio_Value_Over_Time,
                           fmt=['%.5e'],
                           delimiter=",",
                           header="Data",
                           comments='')
    
    np.savetxt('{}_MertonValueOverTime.csv'.format(path_prefix),
                           merton_Portfolio_Value_Over_Time,
                           fmt=['%.5e'],
                           delimiter=",",
                           header="Data",
                           comments='')

    np.savetxt('{}_MertonStratsOverTime.csv'.format(path_prefix),
                           merton_Strats_Over_Time,
                           fmt=['%.5e'],
                           delimiter=",",
                           header="Data",
                           comments='')  

    np.savetxt('{}_NNStratsOverTime.csv'.format(path_prefix),
                           NN_Strats_Over_Time,
                           fmt=['%.5e'],
                           delimiter=",",
                           header="Data",
                           comments='')                     
            

def simStockMultiscale(num_sample, mu, r, alpha_revert, delta_revert, rho_1, rho_2, rho_12, nu_f, nu_s, delta_t, num_time_interval, mf, ms):
        dw_sample = normal.rvs([0, 0, 0], [[delta_t, rho_1 * delta_t, rho_2 * delta_t], 
                                            [rho_1 * delta_t, delta_t, rho_12 * delta_t],
                                            [rho_2 * delta_t, rho_12 * delta_t, delta_t]], size=[num_sample, num_time_interval])
        x_sample = np.zeros([num_sample, num_time_interval + 1])
        x_sample[:, 0] = np.ones(1) * 100
        
        y_sample = np.zeros([num_sample, num_time_interval + 1])
        y_sample[:, 0] = np.ones(1) * -1
        
        z_sample = np.zeros([num_sample, num_time_interval + 1])
        z_sample[:, 0] = np.ones(1) * -1
        
        #print("y_sample before creation: ", y_sample)
        #print("z_sample before creation: ", z_sample)
        
        # validated
        for i in range(num_time_interval):
            vol_factor = np.exp(y_sample[:, i] + z_sample[:, i])
            x_sample[:, i + 1] = x_sample[:, i] * np.exp((r - 0.5 * np.power(vol_factor, 2)) * delta_t + np.multiply(vol_factor, dw_sample[:, i, 0]))
        
            y_sample[:, i + 1] = y_sample[:, i] + alpha_revert * delta_t * (mf - y_sample[:, i]) + nu_f * math.sqrt(2 * alpha_revert) * dw_sample[:, i, 1] 
            z_sample[:, i + 1] = z_sample[:, i] + delta_revert * delta_t * (ms - z_sample[:, i]) + nu_s * math.sqrt(2 * delta_revert) * dw_sample[:, i, 2] 
            
        dw_Factors = np.zeros(shape = (0, 2, num_time_interval))
        process_Factors = np.zeros(shape = (0, 2, num_time_interval + 1))
        
        stockPrice = np.reshape(x_sample, (len(x_sample), 1, num_time_interval + 1))
        #print("ITERATING THRU RESTRUCTURING")
        # VALIDATED RESTRUCTURING SCHEME
        for i in range(num_sample):
            currSample = dw_sample[i]
            #currXSample = x_sample[i]
            currYSample = y_sample[i]
            currZSample = z_sample[i]
    
            #currOne = currSample[:, 0]
            currTwo = currSample[:, 1]
            currThree = currSample[:, 2]
            
            #print("currOne: ", currSample[:, 0])
            #print("currTwo: ", currTwo)
            #print("currThree: ", currThree)
            
            #print("currYSample: ", currYSample)
            #print("currZSample: ", currZSample)
            #print("currOne: ", currOne)    

            tempArrayForDWFactors = np.ndarray(shape = (2, num_time_interval), buffer = np.append(currTwo, currThree))
            #print("temp Array: ", tempArray)
            tempArrayForFactors = np.ndarray(shape = (2, num_time_interval + 1), buffer = np.append(currYSample, currZSample))
    
            dw_Factors = np.append(dw_Factors, np.array([tempArrayForDWFactors]), axis = 0)
            process_Factors = np.append(process_Factors, np.array([tempArrayForFactors]), axis = 0)
            
            #stockPrice = np.append(stockPrice, currXSample, axis = 0)
        #print("dw_Factors: ", dw_Factors)
        #print("process_Factors: ", process_Factors)
        #print("stockPrice: ", stockPrice)
        return dw_Factors, process_Factors, stockPrice


if __name__ == '__main__':
    main()
