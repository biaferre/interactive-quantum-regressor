## Imports

import os
import math
from datetime import datetime
from itertools import product as itertools_product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize as sklearn_normalize
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import jax
from jax import numpy as jnp
#import jax.scipy.linalg


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


def plot_regression_dataset(X, y, y_pred=None, title="Linear Regression Dataset"):
    """
    Plot the regression dataset and optionally the predictions

    Parameters:
    -----------
    X : numpy array
        Input features
    y : numpy array
        True target values
    y_pred : numpy array, optional
        Predicted target values
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))

    # Plot actual data points
    plt.scatter(X, y, alpha=0.6, label='Actual data', color='blue')

    # Plot predictions if provided
    if y_pred is not None:
        # Sort by X for proper line plotting
        sorted_indices = np.argsort(X.flatten())
        plt.plot(X[sorted_indices], y_pred[sorted_indices],
                color='red', linewidth=2, label='Predictions')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



def plot_regression_dataset(X, y, y_pred=None, title="Linear Regression Dataset", y_lim=None):
    """
    Plot the regression dataset and optionally the predictions

    Parameters:
    -----------
    X : numpy array
        Input features
    y : numpy array
        True target values
    y_pred : numpy array, optional
        Predicted target values
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))

    # Plot actual data points
    plt.scatter(X, y, alpha=0.6, label='Actual data', color='blue')

    if y_pred is not None:
      plt.plot(X, y_pred,
            color='red', linewidth=2, label='Predictions')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.show()



def evaluate_model(best_params, X, X_final, y_exact):

  def evaluate(params, data):
    y_pred = jax.vmap(regressor, in_axes=(0,None))(data, params)
    return y_pred

  y_predictions=evaluate(best_params,X_final)
  from sklearn.metrics import r2_score
  r2 = round(float(r2_score(y_exact, y_predictions)),3)
  print("R^2 Score:", r2)
  plot_regression_dataset(X, y_exact, y_predictions,title="Real data vs. Prediction values")
  return r2


def evaluate_plot_model(best_params, X, X_final, y_exact,plot=False):

  def evaluate(params, data):
    y_pred = [regressor(x, params) for x in data]
    return y_pred

  y_predictions=evaluate(best_params,X_final)
  from sklearn.metrics import r2_score
  r2 = round(float(r2_score(y_exact, y_predictions)),3)
  print("R^2 Score:", r2)
  if plot:
    plot_regression_dataset(X, y_exact, y_predictions,title="Real data vs. Prediction values")
  return r2


from sklearn.metrics import r2_score

def evaluate_plot__many_model(list_best_params, X, list_of_X_final, y_exact,plot=True, list_of_functions_names=[], function=""):

  def evaluate(params, data):
    y_pred = [regressor(x, params) for x in data]
    return y_pred

  r2_list = []
  plt.figure(figsize=(10, 6))
  plt.scatter(X, y_exact, alpha=0.6, label='Training data', color='blue')
  
  indexNumber=0
  #for best_params, function_name in zip(list_best_params, list_of_functions_names):
  for best_params, X_final,function_name in zip(list_best_params, list_of_X_final, list_of_functions_names):

    y_predictions=evaluate(best_params,X_final)
    
    r2 = round(float(r2_score(y_exact, y_predictions)),3)
    r2_list.append(r2)

    print("R^2 Score:", r2)
    print("ITERATION INSIDE")
    plt.plot(X, y_predictions, linewidth=2,label=function_name) #color='red', 
    indexNumber+=1
  plt.xlabel('X')
  plt.ylabel('y')
  plt.title("IQR with different encodings - Function "+function)
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.savefig("figs/comparison_IQR_different_encodings_function_"+function+".png")
  plt.show()
  return r2_list


def normalize(x):
    return x / (np.linalg.norm(x) + 1e-16)


def get_p(psi):
    """
    |psi><psi| density matrix.
    """
    psi = np.matrix(psi)
    return psi * psi.getH()



def get_weighted_sigmaQ_jnp(param, iqcpq=False):
    """
    Build sigma_Q.

    - If iqcpq=False: linear combination of Pauli matrices + identity (Eq. 16).
    - If iqcpq=True: builds an n-level Hermitian matrix with fixed diagonal/off-diagonal.
    """
    if iqcpq:
        n = len(param)
        diagonal = jnp.full(n, 1, dtype=complex)
        diagonal[-1] = -jnp.sum(diagonal[:-1])

        off_diagonal = jnp.full((n, n), 1 + 1j, dtype=complex)
        matrix = jnp.zeros((n, n), dtype=complex)
        jnp.fill_diagonal(matrix, diagonal)
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = off_diagonal[i, j]
                matrix[j, i] = jnp.conj(off_diagonal[i, j])
        return matrix

    sigmaX = jnp.array([[0, 1], [1, 0]], dtype=complex)
    sigmaY = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
    sigmaZ = jnp.array([[1, 0], [0, -1]], dtype=complex)
    identity = jnp.array([[1, 0], [0, 1]], dtype=complex)

    sigmaQ = (
        param[0] * sigmaX
        + param[1] * sigmaY
        + param[2] * sigmaZ
        + param[3] * identity
    )
    sigmaq_trace = jnp.trace(sigmaQ)
    #if sigmaq_trace > 0:
    #    return jnp.array(sigmaQ) / sigmaq_trace
    return jnp.array(sigmaQ) / (sigmaq_trace+0.000000000000000000001)
    #return jnp.array(sigmaQ)

def get_weighted_sigmaQ(param, iqcpq=False):
    """
    Build sigma_Q.

    - If iqcpq=False: linear combination of Pauli matrices + identity (Eq. 16).
    - If iqcpq=True: builds an n-level Hermitian matrix with fixed diagonal/off-diagonal.
    """
    if iqcpq:
        n = len(param)
        diagonal = np.full(n, 1, dtype=complex)
        diagonal[-1] = -np.sum(diagonal[:-1])

        off_diagonal = np.full((n, n), 1 + 1j, dtype=complex)
        matrix = np.zeros((n, n), dtype=complex)
        np.fill_diagonal(matrix, diagonal)
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = off_diagonal[i, j]
                matrix[j, i] = np.conj(off_diagonal[i, j])
        return matrix

    sigmaX = np.array([[0, 1], [1, 0]], dtype=complex)
    sigmaY = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigmaZ = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.array([[1, 0], [0, 1]], dtype=complex)

    sigmaQ = (
        param[0] * sigmaX
        + param[1] * sigmaY
        + param[2] * sigmaZ
        + param[3] * identity
    )
    sigmaq_trace = np.trace(sigmaQ)
    if sigmaq_trace > 0:
        return np.array(sigmaQ) / sigmaq_trace
    return np.array(sigmaQ)



def get_sigmaQ_from_polar_coord(param):
    """
    Polar-coordinates version of sigma_Q, normalized to trace 1.
    """
    r, theta, phi = param

    rx = r * np.sin(theta) * np.cos(phi)
    ry = r * np.sin(theta) * np.sin(phi)
    rz = r * np.cos(theta)

    sigmaX = np.array([[0, 1], [1, 0]], dtype=complex)
    sigmaY = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigmaZ = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.array([[1, 0], [0, 1]], dtype=complex)

    return (identity + rx * sigmaX + ry * sigmaY + rz * sigmaZ) / 2.0


def get_sigmaE(vector_x, vector_w, dic_classifier_params, ndse=False):
    """
    Environment operator sigma_E (Eq. 17), JAX-compatible.
    """
    load_inputvector_env_state = dic_classifier_params.get(
        "load_inputvector_env_state", False
    )

    vx = jnp.atleast_1d(vector_x).flatten()
    vw = jnp.atleast_1d(vector_w).flatten()

    if load_inputvector_env_state:
        return jnp.diag(vx)
    # default: element-wise product on diagonal
    return jnp.diag(vx * vw)



import scipy
def get_U_operator_jax(sigmaQ, sigmaE):
    """
    U = exp(+i * (sigma_Q ⊗ sigma_E))  (Eq. 15/22 from the paper).
    Derived from H_int = -ℏg σ_Q⊗σ_E and U(t) = exp(-iH_int t/ℏ).
    """
    H = jnp.kron(sigmaQ, sigmaE)
    U = jax.scipy.linalg.expm(1j * H)
    return U

def get_U_operator(sigmaQ, sigmaE):
    """
    U = exp(+i * (sigma_Q ⊗ sigma_E))  (Eq. 15/22 from the paper).
    Derived from H_int = -ℏg σ_Q⊗σ_E and U(t) = exp(-iH_int t/ℏ).
    """
    H = np.kron(sigmaQ, sigmaE)
    U = scipy.linalg.expm(1j * H)
    return U


def iqc_regressor(
    vector_x,
    vector_ws,
    normalize_x=False,
    normalize_w=False,
    dic_classifier_params=None,
    N_qubits=None,
    N_qubits_tgt=None,
    load_inputvector_env_state=False #Brito et al. Model (amplitude encoding of information)
):
    """
    Core IQC-based regressor (inference path):
    - Builds sigma_Q, sigma_E
    - Evolves ρ_cog ⊗ ρ_env via U
    - Takes partial trace over environment
    - Returns expectation value of Pauli-Z as regressed output.
    """

    bias = vector_ws[0]
    c1 = 1#vector_ws[1]
    c2 = 1#vector_ws[2]
    c3 = 1#vector_ws[3]
    c4 = 1#vector_ws[4]
    vector_ws = vector_ws[1:]

    if dic_classifier_params is None:
        dic_classifier_params = {}

    N = len(vector_x)

    sigma_q_params = dic_classifier_params.get("sigma_q_params", [c1,c2,c3,c4])
    use_polar_coordinates_on_sigma_q = dic_classifier_params.get(
        "use_polar_coordinates_on_sigma_q", False
    )

    if normalize_x:
        vector_x = normalize(vector_x)
    if dic_classifier_params.get("use_exponential_on_input", False):
        vector_x = np.exp(vector_x)

    if use_polar_coordinates_on_sigma_q:
        sigmaQ = get_sigmaQ_from_polar_coord(sigma_q_params)
    else:
        sigmaQ = get_weighted_sigmaQ_jnp(sigma_q_params)

    sigmaQ = jnp.array(sigmaQ, dtype=jnp.complex64)

    vector_x = jnp.array(vector_x, dtype=jnp.float32)

    p_env = jnp.ones((N, 1)) / jnp.sqrt(N)
    p_env = p_env @ p_env.T

    p_cog = jnp.ones((2, 1)) / jnp.sqrt(2)
    p_cog = p_cog @ p_cog.T

    ##vector_ws_jnp = jnp.array(vector_ws, dtype=jnp.float32) #[jnp.array(w, dtype=jnp.float32) for w in vector_ws]
    vector_ws = jnp.array(vector_ws, dtype=jnp.float32) #[jnp.array(w, dtype=jnp.float32) for w in vector_ws]

    p_cog_new = p_cog
    U_operators = []
    p_out = None


    if normalize_w:
        vector_w = vector_w / (jnp.linalg.norm(vector_w) + 1e-16)

    # Equivalent to Eq #15
    if load_inputvector_env_state:

        # We can either keep only weights (in case we have only one environment)
        #sigmaE = jnp.diag(jnp.array(vector_w, dtype=jnp.float32))
        sigmaE = jnp.diag(vector_ws)
    else:
        # Or keep both as the original ICQ article
        #sigmaE = get_sigmaE(vector_x, vector_w, dic_classifier_params)
        sigmaE = jnp.diag(jnp.multiply(vector_x , vector_ws))

    # Eq #19 applied on a Quantum state equivalent of Hadamard(|00...0>) = 1/sqrt(N) * (|00...0> + ... + |11...1>)
    if load_inputvector_env_state:
        # We can either have Hadamard applied to each instance attribute...
        vector_x_norm = (jnp.linalg.norm(vector_x) + 1e-16)

        # env = x1/norm(x) |0> + x2/norm(x) |1> .... + xn/norm(x) |n>
        p_env = jnp.array(vector_x).reshape((N, 1)) / vector_x_norm
        p_env = p_env @ p_env.T


    #sigmaE = jnp.diag(vector_x * vector_w)
    U_operator = get_U_operator_jax(sigmaQ, sigmaE)
    U_operators.append(U_operator)

    p_cog_env = jnp.kron(p_cog_new, p_env)
    p_out = U_operator @ p_cog_env @ jnp.conj(U_operator).T
    p_cog_new = jnp.trace(p_out.reshape([2, N, 2, N]), axis1=1, axis2=3)

    p_cog_new_00 = jnp.real(p_cog_new[0, 0])
    p_cog_new_11 = jnp.real(p_cog_new[1, 1])

    pauli_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    expectation = jnp.real(jnp.trace(p_cog_new @ pauli_z))

    output_dict = {
        "U_operators": U_operators,
        "p_00": p_cog_new_00,
        "p_11": p_cog_new_11,
    }

    #return expectation, p_cog_new_11, output_dict
    return expectation + bias


iqc_regressor_chinese = lambda x, w: iqc_regressor(x,w)
iqc_regressor_brito = lambda x,w:iqc_regressor(x,w,load_inputvector_env_state=True)



def iqc_regressor_numpy(
    vector_x,
    vector_ws,
    normalize_x=False,
    normalize_w=False,
    dic_classifier_params=None,
    N_qubits=None,
    N_qubits_tgt=None,
    load_inputvector_env_state=False #Brito et al. Model (amplitude encoding of information)
):


    """
    Core IQC-based regressor (inference path):
    - Builds sigma_Q, sigma_E
    - Evolves ρ_cog ⊗ ρ_env via U
    - Takes partial trace over environment
    - Returns expectation value of Pauli-Z as regressed output.
    """

    bias = vector_ws[0]
    c1 = 1#vector_ws[1]
    c2 = 1#vector_ws[2]
    c3 = 1#vector_ws[3]
    c4 = 1#vector_ws[4]
    vector_ws = vector_ws[1:]

    if dic_classifier_params is None:
        dic_classifier_params = {}

    N = len(vector_x)

    sigma_q_params = dic_classifier_params.get("sigma_q_params", [c1,c2,c3,c4])
    use_polar_coordinates_on_sigma_q = dic_classifier_params.get(
        "use_polar_coordinates_on_sigma_q", False
    )

    if normalize_x:
        vector_x = normalize(vector_x)
    if dic_classifier_params.get("use_exponential_on_input", False):
        vector_x = np.exp(vector_x)

    if use_polar_coordinates_on_sigma_q:
        sigmaQ = get_sigmaQ_from_polar_coord(sigma_q_params)
    else:
        sigmaQ = get_weighted_sigmaQ(sigma_q_params)

    sigmaQ = np.array(sigmaQ, dtype=np.complex64)

    ##vector_x = jnp.array(vector_x, dtype=jnp.float32)

    p_env = np.ones((N, 1)) / np.sqrt(N)
    p_env = p_env @ p_env.T

    p_cog = np.ones((2, 1)) / np.sqrt(2)
    p_cog = p_cog @ p_cog.T

    ##vector_ws_jnp = jnp.array(vector_ws, dtype=jnp.float32) #[jnp.array(w, dtype=jnp.float32) for w in vector_ws]

    p_cog_new = p_cog
    U_operators = []
    p_out = None


    if normalize_w:
        vector_w = vector_w / (np.linalg.norm(vector_w) + 1e-16)

    # Equivalent to Eq #15
    if load_inputvector_env_state:

        # We can either keep only weights (in case we have only one environment)
        #sigmaE = jnp.diag(jnp.array(vector_w, dtype=jnp.float32))
        sigmaE = np.diag(vector_ws)
    else:
        # Or keep both as the original ICQ article
        #sigmaE = get_sigmaE(vector_x, vector_w, dic_classifier_params)
        sigmaE = np.diag(np.multiply(vector_x , vector_ws))

    # Eq #19 applied on a Quantum state equivalent of Hadamard(|00...0>) = 1/sqrt(N) * (|00...0> + ... + |11...1>)
    if load_inputvector_env_state:
        # We can either have Hadamard applied to each instance attribute...
        vector_x_norm = (np.linalg.norm(vector_x) + 1e-16)

        # env = x1/norm(x) |0> + x2/norm(x) |1> .... + xn/norm(x) |n>
        p_env = np.array(vector_x).reshape((N, 1)) / vector_x_norm
        p_env = p_env @ p_env.T


    #sigmaE = jnp.diag(vector_x * vector_w)
    U_operator = get_U_operator(sigmaQ, sigmaE)
    U_operators.append(U_operator)

    p_cog_env = np.kron(p_cog_new, p_env)
    p_out = U_operator @ p_cog_env @ np.conj(U_operator).T
    p_cog_new = np.trace(p_out.reshape([2, N, 2, N]), axis1=1, axis2=3)

    p_cog_new_00 = np.real(p_cog_new[0, 0])
    p_cog_new_11 = np.real(p_cog_new[1, 1])

    pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    expectation = np.real(np.trace(p_cog_new @ pauli_z))

    output_dict = {
        "U_operators": U_operators,
        "p_00": p_cog_new_00,
        "p_11": p_cog_new_11,
    }

    #return expectation, p_cog_new_11, output_dict
    return expectation + bias

iqc_regressor_chinese_numpy = lambda x, w: iqc_regressor_numpy(x,w)



def func1(X):
  return 0.5*X

def func2(X):
  return 0.5*X**2 - 0.5

def func3(X):
  return 1/10*(0.4*X*X*X + 0.5*X*X + 0.4*X )

def func4(X):
  return 0.2*jnp.sin(2*X)

def func5(X):
  return 0.5*jnp.sin(4*X)

def func6(X):
  return 0.5*jnp.sin(2*X) + 0.8*jnp.cos(4*X)

def func7(X):
  return 0.9*jnp.sin(2*X) + 0.1*jnp.cos(X)

def g(x,k_lim=2):
  sum=0
  for k in range(-k_lim,k_lim+1):
    if k==0:
      ck = 0.1
    else:
      ck = 0.05+1j*0.05
    sum+= ck * np.exp(1j*k*x)
  return sum.real


import jax
from jax import numpy as jnp
import optax
import jax
from jax import numpy as jnp
import optax

regressor = iqc_regressor_chinese #iqc_regressor_brito#

@jax.jit
def mse(vector_ws,vector_x,targets):
    # We compute the mean square error between the target function and the quantum circuit to quantify the quality of our estimator
    return (regressor(vector_x,vector_ws)-jnp.array(targets))**2
@jax.jit
def loss_fn(params, x,targets):
    # We define the loss function to feed our optimizer
    mse_pred = jax.vmap(mse,in_axes=(None, 0,0))(params,x,targets)
    loss = jnp.mean(mse_pred)
    return loss

def calculate_r2_manual(y_true, y_pred):
    # Convert to numpy arrays for easier calculations
    #y_true = np.array(y_true)
    #y_pred = np.array(y_pred)

    # Calculate the mean of the true values
    y_true_mean = jnp.mean(y_true)

    # Calculate Sum of Squared Residuals (SS_res)
    ss_res = jnp.sum((y_true - y_pred)**2)

    # Calculate Total Sum of Squares (SS_tot)
    ss_tot = jnp.sum((y_true - y_true_mean)**2)

    # Handle the case where SS_tot is zero (e.g., all y_true values are the same)
    #if ss_tot == 0:
    #    return 1.0 if ss_res == 0 else 0.0 # Perfect prediction or no variance to explain

    # Calculate R-squared
    r2 = 1 - (ss_res / ss_tot)
    return r2

def loss_fn_r2(params, x,targets):
  y_pred = jax.vmap(regressor, in_axes=(0,None))(x, params)

  return -calculate_r2_manual(targets, y_pred)




def training_model(X=None, X_final=None, y_exact=None, max_steps=1000):
  opt = optax.adamax(learning_rate=0.01)

  @jax.jit
  def update_step_jit(i, args):
      # We loop over this function to optimize the trainable parameters
      params, opt_state, data, targets, print_training = args
      loss_val, grads = jax.value_and_grad(loss_fn)(params, data, targets)
      updates, opt_state = opt.update(grads, opt_state)
      params = optax.apply_updates(params, updates)

      def print_fn():
          jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)
      # if print_training=True, print the loss every 50 steps
      jax.lax.cond((jnp.mod(i, 50) == 0 ) & print_training, print_fn, lambda: None)
      return (params, opt_state, data, targets, print_training)

  @jax.jit
  def optimization_jit(params, data, targets, print_training=False):
      opt_state = opt.init(params)
      args = (params, opt_state, jnp.asarray(data), targets, print_training)
      # We loop over update_step_jit max_steps iterations to optimize the parameters
      (params, opt_state, _, _, _) = jax.lax.fori_loop(0, max_steps+1, update_step_jit, args)
      return params

  n_input_features_final=X_final.shape[1]

  #weights = jnp.ones(n_input_features_final)
  #bias = jnp.array(0.)
  #params = {"weights": weights, "bias": bias}

  key = jax.random.key(10)
  print("qtde params", n_input_features_final)
  params_shape = (n_input_features_final+1,) #FOI RETIRADO O +4, pois o sigmaQ está 1,1,1,1
  params = np.random.random(size=params_shape)#jax.random.uniform(key, shape=params_shape,minval=-50, maxval=50) #np.random.random(size=params_shape)
  best_params=optimization_jit(params, X_final, jnp.array(y_exact), print_training=True)

  return best_params



import numpy
import pyswarms as ps

regressor = iqc_regressor_chinese #iqc_regressor_brito#


def fitness_func(solutions, X_final):
  global y_exact_
  results = []
  #y_pred = jax.vmap(regressor, in_axes=(0,None))(X, solution)
  for solution in solutions:
    #y_pred = [regressor(X,solution) for X in X_final]

    #def evaluate(params, data):
    y_pred = jax.vmap(regressor, in_axes=(0,None))(X_final, jnp.array(solution))
    #return y_pred
  
    #fitness = np.mean((np.array(y_exact)-np.array(y_pred))**2)#-r2_score(y_exact, y_pred)
    fitness = -r2_score(y_exact_, y_pred)
    results.append(fitness)
  return results


def training_model_PSO(X=None, X_final=None, y_exact=None, max_steps=1000, n_particles=10, options_optimizer=None,verbose=False):
  global y_exact_
  y_exact_ = y_exact
  n_input_features_final=X_final.shape[1]+1 #FOI RETIRADO o +4 porque esse modelo
  print("dim+1: ", n_input_features_final) #+1 because of BIAS value

  # Set-up hyperparameters
  # c1: cognitive parameter, c2: social parameter, w: inertia weight
  if options_optimizer is None:
    options_global_best = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    #options_local_best = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}

    options_optimizer = options_global_best


  # Call instance of GlobalBestPSO optimizer
  optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=n_input_features_final, options=options_optimizer)
  #optimizer = ps.single.LocalBestPSO(n_particles=n_particles, dimensions=n_input_features_final, options=options_local_best)

  # Perform optimization
  # The 'sphere' function is a built-in example objective function
  for ITER in range(max_steps):
    best_cost, best_pos = optimizer.optimize(lambda solutions: fitness_func(solutions, X_final), iters=1,verbose=verbose)
    #print("best cost", best_cost)
    if best_cost <= -0.995:
      print("finalizado na iteracao", ITER)
      break

  #best_cost, best_pos = optimizer.optimize(lambda solutions: fitness_func(solutions, X_final), iters=max_steps,verbose=True)

  print(f"Best cost: {best_cost}")
  print(f"Best position (x, y): {best_pos}")

  return best_pos, best_cost

def generate_dataset(x_min=-1, x_max=1, padding_size=1,dataset_size=300, repeat_input=False,func=None):

  X = np.linspace(x_min, x_max, dataset_size).reshape(-1, 1)
  y_exact = func(X)


  # Create a new column of zeros with the same number of rows
  new_column = 0.9*np.ones((X.shape[0], padding_size))

  # Add the new column to the right of the matrix
  # Use axis=1 for column-wise concatenation
  X_final = np.concatenate((X, new_column), axis=1)

  if repeat_input:
    X_final = np.concatenate((X_final,np.repeat(X, repeat_input,axis=1)), axis=1)

  print(X_final.shape[1],)

  import math
  if math.log(X_final.shape[1],2) != int(math.log(X_final.shape[1],2)):
    print("Final dataset is not power of 2")

  return X, X_final, y_exact