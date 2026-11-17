## TODO Change the trajectory of the problem, make so it swing 2 times before up.
import numpy as np
import Modeling as m
import cyipopt
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Initializing global variables

# TimeStep
dt = 0.01

# Dimension of state
N = 6

# Dimension of action
M = 1

# Total simulation time
tf = 5.0

# Total Knot points of the simulation
K = round(tf/dt) + 1 # 101

# Optimization variables

# Dimension of the decision variable
x_dim = K * N + (K - 1) * M

# Dimension of the equality constrains
eq_dim = (K - 1) * N +2 * N

# Initial state
x_init = np.array([0.,np.pi,np.pi,0.,0.,0.]) # shape : (N,)

# Final state
x_target = np.array([0.,0.,0.,0.,0.,0.]) # shape : (N,)

# Model of the system
model = m.DoublePendulumCart()

# System dynamics from Modeling.py (No jax library)

# (state :shape(N,1) action:shape(M,1)) -> state_dot : shape(N,) - It got reshaped, (N,1)
dynamics = lambda x, u : model.dynamics(x.reshape((N,1)),u.reshape((M,1))).reshape((N,))

# Objective function variables

# Gains of the 13 attempt
# The Weight matrix of state
Q = 0.5*np.eye(N)

# The Weight matrix of actions
R = 0.01*np.eye(M)


# Let's evaluate the jacobians with finite differences
def finite_diff(f, z, M_dim, eps = 1e-9):
    N_dim = z.shape[0]
    jac = np.zeros((M_dim, N_dim))
    for i in range(N_dim):
        zp = np.copy(z)
        zp[i] += eps
        zm = np.copy(z)
        zm[i] -= eps
        jac[:, i:i+1] = (f(zp) - f(zm)) / (2. * eps)
    return jac


class TrajOpt:
    def __init__(self):
        pass
    
    # Objective Function for the trajectory optimization
    def objective(self,z):

        # Extracting the state vectors (N,)->(N,1) from decision variable
        states = [z[i*N:(i+1)*N].reshape(N,1) for i in range(K)]
        
        # Extracting the action vectors (M,)->(M,1) from decision variable
        u_controls = [z[K * N + i * M: K * N + i * M + M].reshape(M,1) for i in range(K - 1)]
        
        # Initializing the total cost 1x1 matrix
        total_cost = np.array([[0.0]])
        
        # Calculating the cost
        for i in range(K - 1):
            total_cost += 0.5 * u_controls[i].T @ R @ u_controls[i]
            total_cost += 0.5 * states[i].T @ Q @ states[i]
        
        return total_cost[0,0] # Returning the scalar number of the 1x1 matrix

    def gradient(self,z):

        obj = lambda s: self.objective(s).reshape(1,1)

        return finite_diff(obj,z,1).reshape(x_dim,)


    # Setting up the equality constrains of the optimization problem (Trajectory)
    # Also, updating the constrains.
    def constraints(self, x):
        
        # Initializing the constrain vector
        c = np.zeros((eq_dim,))

        # Iterate K-2 times to pass through all the sequential states and action
        for k in range(K-2):
            
            # Sequential states
            x0 = x[k * N : (k + 1) * N]
            x1 = x[(k + 1) * N : (k + 2) * N]
            
            # Sequential actions
            u = x[K * N + k * M : K * N + k * M + M]
            u1 = x[K * N + (k + 1) * M : K * N + (k + 1) * M + M]
            
            # Setting up the constrain for the next state to be matched
            # x1 : is the reference state that generated
            # x0 : is the reference state that generated
            # x0 + (dt/2) *(dynamics(x0, u) + dynamics(x1,u1)) : is the approximation of the next state using trapezoidal rule
            c[k * N : (k + 1) * N] = x1 - (x0 + (dt/2) *(dynamics(x0, u) + dynamics(x1,u1)))

        # Getting the initial state that generated
        x0 = x[0 : N]
        
        # Getting the final state that generated
        xN = x[(K - 1) * N : K * N]

        # The last 2*N position of the equality constrains
        si = eq_dim - 2 * N
        
        # Setting the pre-last position of states in equality constrains
        # Trying to mach the x0 : generated initial state to x_init : actual initial state
        c[si : si + N] = x0 - x_init
        
        # Setting the last position of states in equality constrains
        # Trying to mach the xN : generated last state to x_target : actual last state
        c[si + N : ] = (xN - x_target)

        return c # Returning the equality constrains vectors (eq_dim,)
    
    def jacobian(self,x):

        const = lambda z: self.constraints(z.reshape(x_dim,)).reshape(eq_dim,1)

        return finite_diff(const,x.reshape(x_dim,1),eq_dim)

    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        # Example for the use of the intermediate callback.
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
        
def main():
    
    # 1********** Uncomment to see live result ***********************************************************
    # Initializing the decision variable with zeros
    x0 = np.zeros((x_dim,))

    # Setting a guessed trajectory to warm up the solver
    for k in range(K):
      
        # Weight of the state
        alpha = k / (K - 1)
      
        # Next state is a weight to initial and final state
        # e.g close to initial state then the next state difference more from target than initial.
        x0[k * N : (k + 1) * N] = (1 - alpha) * x_init + alpha * x_target
    
    # Setting up the guessed control for its knot point in trajectory
    # We set them to be zero
    for k in range(K - 1):
        x0[K * N + k * M : K * N + k * M + M] = 0.

    # Setting the lower and upper bound of the decision variable.
    
    # No limits for its state in decision variable
    lb = [None] * x_dim
    ub = [None] * x_dim

    # Add u bounds
    max_control_force :float = 20.0
    for k in range(K - 1):
        lb[K * N + k * M : K * N + k * M + M] = [-max_control_force for _ in range(M)]
        ub[K * N + k * M : K * N + k * M + M] = [max_control_force for _ in range(M)]

    # All constraints are equality constraints
    # Setting to zero cause we want the reference match the prediction from trapezoidal rule
    cl = [0.] * eq_dim
    cu = [0.] * eq_dim


    # Setting up the optimizer 
    nlp = cyipopt.Problem(n=len(x0), m=len(cl), problem_obj=TrajOpt(), lb=lb, ub=ub, cl=cl, cu=cu)

    # nlp.add_option('jacobian_approximation','exact')
    # nlp.add_option('jacobian_approximation','finite-difference-values')
    nlp.add_option('tol', 1e-6) # Convergence tolerance 
    nlp.add_option('max_iter', 100) # Maximum number of iterations
    nlp.add_option("print_level", 3)
    nlp.add_option("nlp_scaling_method", "none")
    nlp.addOption("warm_start_init_point", "yes")
    nlp.addOption("warm_start_bound_push", 100*(1e-6))
    nlp.addOption("warm_start_mult_bound_push", 100*(1e-6))
            


    # Solve the problem
    x, info = nlp.solve(x0)
    
    # 1********************************************************************************************************
    
    
    # 2******* Uncomment to save the file *********************************************************************
    # Saving the solution to a txt file
    np.savetxt('arrayNew.txt', x, fmt='%1.2f', delimiter=', ')
    # 2********************************************************************************************************


    # Loading from existing file the optimized vector
    
    # 3********** Uncomment to see ready example ***************************************************************
#    loaded_array = np.loadtxt('array13.txt', delimiter=',')
#    x = np.copy(loaded_array)
    # 3*********************************************************************************************************
    

    # Graphing the result of the position
    print("Plotting the results")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # ax.plot([k * dt for k in range(K)], [x[k * N + 0] for k in range(K)], label="x")
    # ax.plot([k * dt for k in range(K)], [x[k * N + 1] for k in range(K)], label="θ1")
    # ax.plot([k * dt for k in range(K)], [x[k * N + 2] for k in range(K)], label="θ2")

    # ax.plot([k * dt for k in range(K)], [x[k * N + 3] for k in range(K)], label="$\dot{x}$")
    # ax.plot([k * dt for k in range(K)], [x[k * N + 4] for k in range(K)], label="$\dot{θ1}$")
    # ax.plot([k * dt for k in range(K)], [x[k * N + 5] for k in range(K)], label="$\dot{θ2}$")

    ax.plot([k * dt for k in range(K - 1)], [x[K * N + k * M + 0] for k in range(K - 1)], label="u")

    plt.xlabel("Time")

    plt.legend()
    plt.show()
    
    print("Close the window to show the trajectory")
    
    print("Visualizing the trajectory")
    
    # Extracting the states and actions from decision variable
    states = [x[i*N:(i+1)*N].reshape(N,1) for i in range(K)]
    u_controls = [x[K * N + i * M: K * N + i * M + M].reshape(M,1) for i in range(K - 1)]

    # u_controls= [np.zeros((M,1)) ] + u_controls

    visual :m.Visualization = m.Visualization("Double_Pendulum_Cart\nTrajectory Visualization","x","y",[-5,9],[-2,2],True)
    visual.fig.canvas.mpl_connect('close_event',visual._on_close)

    rect = patches.Rectangle((0.0 - 0.3, 0.0 - 0.15), 0.6, 0.3, linewidth=0.5, edgecolor='blue', facecolor='blue')
    visual.axes.add_patch(rect)

    lineDP, = visual.axes.plot([],[],"o-",markersize = 7,lw = 2)
    lineDP.set_data([],[])

    # Visualization loop
    for i in range(K-1):

        x_new = np.copy(states[i])

        functions = [(visual.move_rectangle,(rect,x_new[0,0],0.0))
                    ,(visual.move_double_pendulum,(x_new[1,0],x_new[2,0],np.array([[x_new[0,0],0.0]]).T,model.l2,model.l3,lineDP))
                    ]

        visual.update(functions)
        
        # Trace of the end effector - the q2 coordinates of double pendulum
        set_zero_offset = np.pi/2
        q1 = x_new[1,0]
        q2 = x_new[2,0]
        l1 = model.l2
        l2 = model.l3
        x2 = l1 * np.cos(q1 + set_zero_offset) + l2 * np.cos(q2 + set_zero_offset)
        y2 = l1 * np.sin(q1 + set_zero_offset) + l2 * np.sin(q2 + set_zero_offset)
        visual.axes.scatter(x2 + x_new[0,0],y2,s=10,c="red",alpha = 0.4)
        
    plt.ioff()
    plt.show()
    
    return 0

if __name__ == "__main__":
    main()
