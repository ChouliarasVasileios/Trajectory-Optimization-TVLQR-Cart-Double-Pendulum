import numpy as np
import jax.numpy as jnp
import jax
import Modeling_jax as m
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TVLQR:
    
    def __init__(self,model :m.DoublePendulumCart,N :int,M :int,knots :int,trajectory_states,u_controls,dt = 0.05):
        
        # Modeling of the system
        self.model = model
        
        # State Dimension
        self.N = N
        
        # Action Dimension
        self.M = M
        
        # Number of Knot Points
        self.knots = knots
        
        # The generated states of trajectory
        self.trajectory_states = trajectory_states
        
        # The generated control of trajectory
        self.u_controls = u_controls
        
        # StepTime
        self.dt = dt
    
    # Linearize around a state and action
    def linearize(self,xk = jnp.array([0.,0.,0.,0.,0.,0.]), uk = jnp.array([0.0])):
        
        # Discretization of the continues dynamics
        f_discrete = lambda x,u:(self.model.runge_kutta4(x.reshape(self.N,1),u.reshape(self.M,1),self.dt)).reshape(self.N,)
        # print(f_discrete(xk,uk),jnp.shape(f_discrete(xk,uk)))
        
        # Tensor derivative of f_discrete with respect to x:state
        df_discrete_tensor_wrt_x = jax.jit(jax.jacobian(f_discrete,0))
        
        # The Jacobian of discrete dynamics wrt x == Ad
        df_discrete_wrt_x = lambda x,u: df_discrete_tensor_wrt_x(x,u)#.reshape(self.N,self.N)
        # print(df_discrete_wrt_x(xk,uk),jnp.shape(df_discrete_wrt_x(xk,uk)))
        
        # Tensor derivative of f_discrete with respect to u:controls
        df_discrete_tensor_wrt_u = jax.jit(jax.jacobian(f_discrete,1))
        
        # The Jacobian of discrete dynamics wrt u == bd
        df_discrete_wrt_u = lambda x,u: df_discrete_tensor_wrt_u(x,u)#.reshape(self.N,self.M)
        # print(df_discrete_wrt_u(xk,uk),jnp.shape(df_discrete_wrt_u(xk,uk)))
        
        # Ad and Bd respectively
        # print(df_discrete_wrt_x(xk,uk))
        # print(df_discrete_wrt_u(xk,uk))
        return df_discrete_wrt_x(xk,uk),df_discrete_wrt_u(xk,uk)
    
    # Riccati Recurion for Time-Varying Linear Quadratic Regulator
    def riccati_recursion_TVLQR(self,Q,QN,R,Ad,Bd):
                
        P = [np.zeros((self.N,self.N))] * self.knots
        K = [np.zeros((self.M,self.N))] * (self.knots - 1)
        
        P[self.knots-1] = QN
        
        for i in range(self.knots - 2, -1, -1):
            
            K[i] = np.linalg.inv(R + Bd[i].T @ P[i+1] @ Bd[i]) @ Bd[i].T @ P[i+1] @ Ad[i]
            P[i] = Q + Ad[i].T @ P[i+1] @ (Ad[i] - Bd[i] @ K[i])
        
        return P,K
    
    def my_controller(self,current_state,Km,i):
        return np.clip(self.u_controls[i] - Km@(current_state - self.trajectory_states[i]),-20.0,20.0)
    
    # # Controller for the generated trajectory 
    # def my_controller(self,current_state,Ks):
    #     u_controller = np.array([[0.]])
    #     error_state = np.zeros((self.N,1))
    #     minimum_distance :float = np.inf
    #     index :int = -1
    #     for i in range(self.knots-1):
    #         error_state  = (current_state[0:3] - self.trajectory_states[i][0:3])
    #         distance = np.linalg.norm(error_state)
            
    #         if distance < minimum_distance:
    #             minimum_distance = distance
    #             index = i
        
    #     u_controller = self.u_controls[index] - Ks[index]@(current_state - self.trajectory_states[index])
    #     # u_pd = np.array([[0.05,0.05,0.05,0.015,0.025,0.025]])@(self.trajectory_states[-1] - current_state)
    #     return np.clip(u_controller,-20.0,20.0)
    
def main():
    
    # Knots Points
    K = 101
    
    # Dimension of state
    N = 6
    
    # Dimension of action
    M = 1
    
    # The system modeling
    model = m.DoublePendulumCart()
    
    # Loading the trajectory generated
    loaded_array = np.loadtxt('arrayNew.txt', delimiter=',')
    x = np.copy(loaded_array)
    
    # Extracting the state and actions from generated trajectory
    states = [x[i*N:(i+1)*N] for i in range(K)]    
    u_controls = [x[K * N + i * M: K * N + i * M + M] for i in range(K - 1)]
    
    # Initializing the Time - Varying LQR
    tvlqr = TVLQR(model,N,M,K,None,None)
    
    # Initializing the linearized matrixes for each trajectory Knot
    As_d = [np.zeros((N, N))]*(K-1)
    Bs_d = [np.zeros((N, M))]*(K-1)
    
    # Linearize around the trajectory
    for i in range(K-1):
        Ad,Bd = tvlqr.linearize(xk = states[i],uk = u_controls[i])
        # print(np.shape(Ad))
        # print(np.shape(Bd))
        # exit(1)
        As_d[i] = Ad
        Bs_d[i] = Bd
    
    # Initializing the riccati parameter
    Q = QN = 18000*np.eye(N) # affect state
    R = 0.05*np.eye(M) # affect action
    
    # Q = QN = np.eye(N) # affect state
    # R = 0.5*np.eye(M) # affect action
    
    
    # The gain matrixes for each Knot point
    Ps,Ks = tvlqr.riccati_recursion_TVLQR(Q,QN,R,As_d,Bs_d)
    
    # Initializing the graphics
    visual :m.Visualization = m.Visualization("Double_Pendulum_Cart","x","y",[-5,15],[-2,2],True)
    visual.fig.canvas.mpl_connect('close_event',visual._on_close)
    
    rect = patches.Rectangle((0.0 - 0.3, 0.0 - 0.15), 0.6, 0.3, linewidth=0.5, edgecolor='blue', facecolor='blue')
    visual.axes.add_patch(rect)
    
    lineDP, = visual.axes.plot([],[],"o-",markersize = 7,lw = 2)
    lineDP.set_data([],[])
    
    # Initializing the state and controls of generated trajectory to tvlqr object
    states = [x[i*N:(i+1)*N].reshape(N,1) for i in range(K)]
    tvlqr.trajectory_states = states
    u_controls = [x[K * N + i * M: K * N + i * M + M].reshape(M,1) for i in range(K - 1)]
    tvlqr.u_controls = u_controls
    
    
    # Initial positions
    x_init = np.copy(states[0].reshape(N,1))
    current_state = np.copy(x_init)
    
    # For open loop simulation
    for k in range(K-1):
        u_controller = tvlqr.my_controller(current_state,Ks[k],k)
        print(u_controller)
        # Advance the system
        current_state = model.runge_kutta4(current_state,u_controller)
        
        # Updating the graphics
        functions = [(visual.move_rectangle,(rect,current_state[0,0],0.0))
                    ,(visual.move_double_pendulum,(current_state[1,0],current_state[2,0],np.array([[current_state[0,0],0.0]]).T,model.l2,model.l3,lineDP))
                    ]
        visual.update(functions)
        # Trace of the end effector - the q2 coordinates of double pendulum
        set_zero_offset = np.pi/2
        q1 = current_state[1,0]
        q2 = current_state[2,0]
        l1 = model.l2
        l2 = model.l3
        x2 = l1 * np.cos(q1 + set_zero_offset) + l2 * np.cos(q2 + set_zero_offset)
        y2 = l1 * np.sin(q1 + set_zero_offset) + l2 * np.sin(q2 + set_zero_offset)
        visual.axes.scatter(x2 + current_state[0,0],y2,s=10,c="red",alpha = 0.4)
    
    # For closed loop simulation
    # Real World Simulation
    # it = 0
    # while True:
    #     # Calculating the control input
    #     u_controller = tvlqr.my_controller(current_state,Ks)
        
    #     # Advance the system
    #     current_state = model.runge_kutta4(current_state,u_controller)
        
    #     # Updating the graphics
    #     functions = [(visual.move_rectangle,(rect,current_state[0,0],0.0))
    #                 ,(visual.move_double_pendulum,(current_state[1,0],current_state[2,0],np.array([[current_state[0,0],0.0]]).T,model.l2,model.l3,lineDP))
    #                 ]
        
    #     visual.update(functions)
    #     # Trace of the end effector - the q2 coordinates of double pendulum
    #     set_zero_offset = np.pi/2
    #     q1 = current_state[1,0]
    #     q2 = current_state[2,0]
    #     l1 = model.l2
    #     l2 = model.l3
    #     x2 = l1 * np.cos(q1 + set_zero_offset) + l2 * np.cos(q2 + set_zero_offset)
    #     y2 = l1 * np.sin(q1 + set_zero_offset) + l2 * np.sin(q2 + set_zero_offset)
    #     visual.axes.scatter(x2 + current_state[0,0],y2,s=10,c="red",alpha = 0.4)
        
    #     # Calculating the error
    #     error_norm = np.linalg.norm(current_state - states[-1])
    #     print(error_norm)
    #     if error_norm < 1e-1:
    #         break
        
    #     # Task Failed
    #     if it > 2000:
    #         break
    #     it += 1
    print("Simulation ended")
    
    plt.ioff()
    plt.show()
    
    return 0

if __name__  == "__main__":
    main()