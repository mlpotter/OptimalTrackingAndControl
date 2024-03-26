from src_range_publish.control.Sensor_Dynamics import *

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_option = 1

    dt = 0.1

    if test_option == 0:
        p = jnp.array([[0, 0,0]])
        chi = jnp.array([[0]])

        unicycle_state = jnp.column_stack((p, chi))

        # test to see if input affects output in realistic manner?
        av_inputs = jnp.array([jnp.pi/2,jnp.pi/2,jnp.pi/2,0])
        v_inputs = jnp.array([1,2,3,4])
        U = jnp.column_stack((v_inputs,av_inputs))

        unicycle_states = unicycle_kinematics_single_integrator(U,unicycle_state,dt)


        plt.figure()
        plt.plot(unicycle_states[..., 0], unicycle_states[..., 1], 'r*')
        plt.plot(unicycle_states[...,0],unicycle_states[...,1],'r.-')
        plt.show()

        print(unicycle_states)

        av_inputs = jnp.array([jnp.pi/2]*25)
        v_inputs = jnp.array([5]*25)
        U = jnp.column_stack((v_inputs,av_inputs))

        unicycle_states = unicycle_kinematics_single_integrator(U,unicycle_state,dt)


        plt.figure()
        plt.plot(unicycle_states[..., 0], unicycle_states[..., 1], 'r*')
        plt.plot(unicycle_states[...,0],unicycle_states[...,1],'r.-')
        plt.axis('equal')
        plt.show()

        print(unicycle_states)

        av_inputs = jnp.array([jnp.pi/2,jnp.pi/2,jnp.pi/2,jnp.pi/2,jnp.pi/2,-jnp.pi/2,-jnp.pi/2,-jnp.pi/2,-jnp.pi/2,-jnp.pi/2])
        v_inputs = jnp.array([5]*5 + [10]*5)
        U = jnp.column_stack((v_inputs,av_inputs))

        unicycle_states= unicycle_kinematics_single_integrator(U,unicycle_state,dt)


        plt.figure()
        plt.plot(unicycle_states[..., 0], unicycle_states[..., 1], 'r*')
        plt.plot(unicycle_states[...,0],unicycle_states[...,1],'r.-')
        plt.axis('equal')
        plt.show()

        print(unicycle_states)

    if test_option == 1:

        p = jnp.array([[0.0, 0.0,0.0]])
        chi = jnp.array([[0.0]])
        v = jnp.array([[0.0]]);
        av = jnp.array([[0.0]]);

        unicycle_state = jnp.column_stack((p, chi,v,av))

        dt = 0.1

        # test to see if input affects output in realistic manner?
        a_inputs = jnp.array([35.0]*15)
        aa_inputs = jnp.array([0]*15)

        U = jnp.column_stack((a_inputs, aa_inputs))

        unicycle_states = unicycle_kinematics_double_integrator(U, unicycle_state, dt)

        plt.figure()
        plt.plot(unicycle_states[0,0, 0], unicycle_states[0,0, 1], 'r*')
        plt.plot(unicycle_states[..., 0].ravel(), unicycle_states[..., 1].ravel(), 'r.-')
        plt.show()

        print(jnp.round(unicycle_states,2))

        p = jnp.array([[0, 0]])
        chi = jnp.array([[0]])
        v = jnp.array([[0]]);
        av = jnp.array([[0]]);

        a_inputs = jnp.array([-1]*10+[-1]*3)
        aa_inputs = jnp.array([-jnp.pi/4]*5 + [-jnp.pi/4]*5 + [0]*3)
        U = jnp.column_stack((a_inputs,aa_inputs))

        unicycle_states = unicycle_kinematics_double_integrator(U,unicycle_state,dt)


        plt.figure()
        plt.plot(unicycle_states[0,0, 0], unicycle_states[0,0, 1], 'r*')
        plt.plot(unicycle_states[..., 0].ravel(), unicycle_states[..., 1].ravel(), 'r.-')
        plt.axis('equal')
        plt.show()

        print(jnp.round(unicycle_states,2))

        print("Check accel Limits")
        a_inputs = jnp.array([50.0, 50.0, -50.0, -50.0,-50])
        aa_inputs = jnp.array([0.0, 0.0, 0.0, 0.0,0])

        U = jnp.column_stack((a_inputs, aa_inputs))

        unicycle_states = unicycle_kinematics_double_integrator(U, unicycle_state, dt=1.0)

        print(jnp.round(unicycle_states,2))

    # apply dynamics to more than one sensor at a time with vmap...
    if test_option == 2:
        from jax import vmap

        dt = 1

        print("TEST OPTION 2")
        p = jnp.array([[0, 0,0],[2,2,2],[3,3,3]])
        chi = jnp.array([[jnp.pi],[jnp.pi/2],[0]])
        v = jnp.array([[0],[0],[0]]);
        av = jnp.array([[0],[0],[0]]);

        unicycle_state = jnp.column_stack((p, chi,v,av))


        dt = 0.1

        a_inputs1 = jnp.array([2*jnp.pi, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi]*4)
        a_inputs2 = jnp.array([2*jnp.pi, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi]*4)
        a_inputs3 = jnp.array([2*jnp.pi, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi]*4)

        aa_inputs1 = jnp.array([10, 20, 30, 40]*4)
        aa_inputs2 =  jnp.array([10, 20, 30, 40]*4)
        aa_inputs3 =  jnp.array([10, 20, 30, 40]*4)

        u1 = jnp.column_stack((a_inputs1,aa_inputs1))
        u2 = jnp.column_stack((a_inputs2,aa_inputs2))
        u3 = jnp.column_stack((a_inputs3,aa_inputs3))

        U = np.stack((u1,u2,u3),axis=0)

        outputs = unicycle_kinematics_double_integrator(U,unicycle_state,dt)


        plt.figure()
        for n in range(p.shape[0]):
            plt.plot(outputs[n,:,0],outputs[n,:,1],'r*--')
        plt.axis('equal')
        plt.show()

    if test_option == 3:
        from jax import vmap

        dt = 1

        print("TEST OPTION 2")
        p = jnp.array([[0, 0,0],[2,2,2],[3,3,3]])
        chi = jnp.array([[jnp.pi],[jnp.pi/2],[0]])

        unicycle_state = jnp.column_stack((p, chi))


        dt = 0.1

        a_inputs1 = jnp.array([2*jnp.pi, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi]*4)
        a_inputs2 = jnp.array([2*jnp.pi, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi]*4)
        a_inputs3 = jnp.array([2*jnp.pi, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi]*4)

        aa_inputs1 = jnp.array([10, 20, 30, 40]*4)
        aa_inputs2 =  jnp.array([10, 20, 30, 40]*4)
        aa_inputs3 =  jnp.array([10, 20, 30, 40]*4)

        u1 = jnp.column_stack((a_inputs1,aa_inputs1))
        u2 = jnp.column_stack((a_inputs2,aa_inputs2))
        u3 = jnp.column_stack((a_inputs3,aa_inputs3))

        U = np.stack((u1,u2,u3),axis=0)

        outputs = unicycle_kinematics_single_integrator(U,unicycle_state,dt)


        plt.figure()
        for n in range(p.shape[0]):
            plt.plot(outputs[n,:,0],outputs[n,:,1],'r*--')
        plt.axis('equal')
        plt.show()