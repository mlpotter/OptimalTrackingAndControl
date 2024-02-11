from src.control.Sensor_Dynamics import *

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_option = 2

    p = jnp.array([[0,0]])
    chi = jnp.array([[0]])
    time_step_size = 1

    # test to see if input affects output in realistic manner?
    if test_option == 0:
        av_inputs = jnp.array([jnp.pi/2,jnp.pi/2,jnp.pi/2,0])
        v_inputs = jnp.array([1,2,3,4])

        plt.figure()
        plt.plot(p[0, 0], p[1, 0], 'r*')

        for v,av in zip(v_inputs,av_inputs):
            chi_next = angle_update(chi,av,time_step_size)
            p_next = position_update(p,v,av,chi,chi_next,time_step_size)
            plt.plot(p_next[0,0].item(),p_next[0,1].item(),'r*')
            plt.quiver(p[0,0].item(),p[0,1].item(),p_next[0,0].item(),p_next[0,1].item())
            p = p_next
            chi = chi_next

        print(chi_next)
        print(p_next)
        plt.show()

    # test if for loop for angles and position work together
    if test_option == 1:
        av_inputs = jnp.array([jnp.pi / 2, jnp.pi / 2, jnp.pi / 2, 0])
        v_inputs = jnp.array([1, 2, 3, 4])
        U = jnp.vstack((v_inputs,av_inputs))
        p_new,chi_new,_,_ = state_multiple_update(p,  U, chi, time_step_size)
        print(chi_new)
        print(p_new)

    # apply dynamics to more than one sensor at a time with vmap...
    if test_option == 2:
        from jax import vmap

        time_step_size = 1

        print("TEST OPTION 4")
        p = jnp.array([[[0, 0]],[[2,2]],[[3,3]]])
        chi = jnp.array([[[0]],[[0]],[[0]]])

        time_step_sizes = jnp.tile(time_step_size,(p.shape[0],1,1))

        av_inputs1 = jnp.array([jnp.pi / 2, jnp.pi / 2, jnp.pi / 2, 0])
        av_inputs2 = jnp.array([jnp.pi / 2, jnp.pi / 2, jnp.pi / 2, 0])
        av_inputs3 = jnp.array([jnp.pi / 2, jnp.pi / 2, jnp.pi / 2, 0])

        v_inputs1 = jnp.array([1, 2, 3, 4])
        v_inputs2 =  jnp.array([1, 2, 3, 4])
        v_inputs3 =  jnp.array([1, 2, 3, 4])

        u1 = jnp.vstack((v_inputs1,av_inputs1))
        u2 = jnp.vstack((v_inputs2,av_inputs2))
        u3 = jnp.vstack((v_inputs3,av_inputs3))

        U = jnp.vstack((jnp.expand_dims(u1,0),jnp.expand_dims(u2,0),jnp.expand_dims(u3,0)))

        print(U)

        outputs = vmap(state_multiple_update,(0,0,0,0))(p,U,chi,time_step_sizes)
        print(outputs[0])

        plt.figure()
        for n in range(p.shape[0]):
            plt.plot(outputs[2][n,:,0,0],outputs[2][n,:,0,1],'r*--')
        plt.show()