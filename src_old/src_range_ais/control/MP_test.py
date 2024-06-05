from src_range.control.MP import *
from src_range.control.Sensor_Dynamics import state_multiple_update

import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
from itertools import combinations

def generate_MP(steps,stds):
    baselines = []

    avs = jnp.linspace(stds[1,0],stds[1,1],10).reshape(-1,1)

    vs = jnp.linspace(stds[0,0],stds[0,1],10).reshape(-1,1)


    av_baselines = jnp.ones((1,steps)) * avs

    v_baselines = jnp.ones((1,steps)) * vs

    m1,n1 = v_baselines.shape
    m2,n2 = av_baselines.shape


    mps = jnp.zeros((m1,m2,n1+n2))
    mps = mps.at[:,:,:n1].set(v_baselines[:,None,:])
    mps = mps.at[:,:,n1:].set(av_baselines)

    mps = mps.reshape(-1,steps,2,order="F")

    return mps


if __name__ == "__main__":

    test_option = 0
    stds = jnp.array([[-10,10],
                      [-90* jnp.pi/180, 90 * jnp.pi/180]])

    p = jnp.array([[0,0]])
    chi = jnp.array([[0]])
    time_step_size = 0.1
    mps = generate_MP(5,stds)
    # test to see if input affects output in realistic manner?
    if test_option == 0:
        av_inputs = jnp.array([jnp.pi/2,jnp.pi/2,jnp.pi/2,0])
        v_inputs = jnp.array([1,2,3,4])
        U = jnp.column_stack((v_inputs,av_inputs))



        ps,chis,positions,ang_velocities = state_multiple_update(p,U,chi,time_step_size)


        plt.figure()
        plt.plot(p[0, 0], p[1, 0], 'r*')
        plt.plot(positions[:,0],positions[:,1],'r.-')
        plt.show()

        print(positions)
        print(ang_velocities)

        av_inputs = jnp.array([jnp.pi/2]*25)
        v_inputs = jnp.array([5]*25)
        U = jnp.column_stack((v_inputs,av_inputs))

        ps,chis,positions,ang_velocities = state_multiple_update(p,U,chi,time_step_size)