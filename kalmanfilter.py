import copy
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy.random import normal

A = np.array(
  [[1, 1],
   [0, 1]], dtype=np.float32)

H = np.array(
  [[1, 0],
   [0, 1]], dtype=np.float32)

X_real = np.array(
  [[0],[1]], dtype=np.float32)

I = np.array(
  [[1, 0],
   [0, 1]], dtype=np.float32)

# Covariance of the process noise
Q = np.array(
  [[0.1, 0], [0, 0.1]], dtype=np.float32)

# Covariance of the measure noise
R = np.array(
  [[1., 0], [0, 1.]], dtype=np.float32)

# Initialize the covariance of the error between 
# evaluated value and the real value.
P = np.array(
  [[1, 0],[0, 1]], dtype=np.float32)

pos_real_list = []
vel_real_list = []
measure_pose_list = []
measure_vel_list = []
eval_pos_list = []
eval_vel_list = []

STEP=20

for i in range(STEP):
  W = normal(loc=0.0, scale=0.316, size=(2, 1))
  V = normal(loc=0.0, scale=1.0, size=(2, 1))
  X_pre_eval = np.dot(A, X_real) 
  X_real = np.dot(A, X_real) + W
  Z = np.dot(H, X_real) + V

  Pk_1 = copy.deepcopy(P)
  P_pre_k = np.dot(np.dot(A, Pk_1), A.transpose()) + Q
  K = np.dot(np.dot(P_pre_k, H.T), inv(np.dot(np.dot(H, P_pre_k), H.T) + R))
  X_eval = X_pre_eval + np.dot(K, Z - np.dot(H, X_pre_eval))

  P = np.dot(P_pre_k, I - np.dot(K, H))

  pos_real_list.append(X_real[0, 0])
  vel_real_list.append(X_real[1, 0])
  measure_pose_list.append(Z[0, 0])
  measure_vel_list.append(Z[1, 0])
  eval_pos_list.append(X_eval[0, 0])
  eval_vel_list.append(X_eval[1, 0])


plt.plot([i for i in range(STEP)], pos_real_list, label="pos_real")
plt.plot([i for i in range(STEP)], measure_pose_list, label='pos_mea')
plt.plot([i for i in range(STEP)], eval_pos_list, label='eval_pos')
plt.legend()

plt.figure()
plt.plot([i for i in range(STEP)], vel_real_list, label='vel_real')
plt.plot([i for i in range(STEP)], measure_vel_list, label='vel_mea')
plt.plot([i for i in range(STEP)], eval_vel_list, label='eval_vel')

plt.legend()
plt.show()