import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Callable


class DoublePendulumCart:
    
    def __init__(self, m1 :float = 0.5, m2 :float = 0.5,m3 :float = 0.5,l2 :float = 0.6,l3 :float = 0.6):
        
        # Cart mass
        self.m1 :float = m1
        
        # First pendulum mass
        self.m2 :float = m2
        
        # Second pendulum mass
        self.m3 :float = m3

        # The first link that connect the cart and q2 angle of  double pendulum
        self.l2 :float = l2

        # The second link that connect the q2 and q3 angles of double pendulum
        self.l3 :float= l3



    # Calculating the continues dynamics of a system
    # b : coefficient of friction
    # g : gravity acceleration
    def dynamics(self,x :ndarray, u :ndarray, b:float = 0.1, g :float = 9.81)  -> ndarray:

        # Unpacking the state vector
        # x1 :float = x[0,0] # Not need to calculate the dynamics
        q2 :float = x[1,0]
        q3 :float = x[2,0]

        x1_dot :float = x[3,0]
        q2_dot :float = x[4,0]
        q3_dot :float = x[5,0]

        # Unpacking the control vector
        u :float = u[0,0]

        # Calculating the angular moments of inertia
        I2 :float = self._angular_moments(self.m2,self.l2)
        I3 :float = self._angular_moments(self.m3,self.l3)


        # Constructing the dynamics equation that need to solve
        
        # First row of the A matrix
        A00 :float = self.m1 + self.m2 + self.m3
        A01 :float = -0.5 * ( self.m2 + 2 * self.m3 ) * self.l2 * np.cos(q2)
        A02 :float = -0.5 * ( self.m3 * self.l3 * np.cos(q3) )
        
        # Second row of the A matrix
        A10 :float = -0.5 * ( self.m2 + self.m3 ) * self.l2 * np.cos(q2)
        A11 :float = self.m3 * ( self.l2**2 ) + I2 + 0.25 * self.m2 * ( self.l2**2 )
        A12 :float = 0.5 * self.m3 * self.l2 * self.l3 * np.cos(q2 - q3)

        # Third row of the A matrix
        A20 :float = A02
        A21 :float = A12
        A22 :float = 0.25 * self.m2 * self.l3**2 + I3

        A :ndarray = np.array([[A00,A01,A02],[A10,A11,A12],[A20,A21,A22]])

        # Constructing the vector of the dynamics equation
        a1 :float = u - b * x1_dot - 0.5 * ( self.m2 + 2 * self.m3 ) * self.l2 * q2_dot**2 * np.sin(q2) -0.5 * self.m3 * self.l3 * q3_dot**2 * np.sin(q3)
        a2 :float = ( 0.5 * self.m2 + self.m3 ) * self.l2 * g * np.sin(q2) - 0.5 * self.m3 * self.l2 * self.l3 * q3_dot**2 * np.sin(q2-q3)
        a3 :float = 0.5 * self.m3 * self.l3 * ( g * np.sin(q3) + self.l2 * q2_dot**2 * np.sin(q2-q3) )

        a :ndarray = np.array([[a1,a2,a3]]).T

        # Solving the dynamic system to calculate the accelerations
        accelerations :ndarray = np.linalg.solve(A,a)

        x1_ddot :float = accelerations[0,0]
        q2_ddot :float = accelerations[1,0]
        q3_ddot :float = accelerations[2,0]

        # Calculating the derivative of state
        x_dot :ndarray = np.array([[x1_dot,q2_dot,q3_dot,x1_ddot,q2_ddot,q3_ddot]]).T 

        return x_dot

    def _angular_moments(self,m :float,l:float) -> float:
        return (1/12) * m * (l**2)
    
    # The discretization of the system using Runge and Kutta 4 order
    # dt : disctretization step Zero Order Hold - post
    def runge_kutta4(self, x : ndarray, u : ndarray, dt :float = 0.05) -> ndarray:
        f1 = self.dynamics(x, u)
        f2 = self.dynamics(x + f1 *  (dt / 2), u)
        f3 = self.dynamics(x + f2 * (dt / 2), u)
        f4 = self.dynamics(x + f3 * dt, u)
        return x + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)




class Visualization:

    def __init__(self,title :str,xlabel :str, ylabel :str, xlim :list, ylim :list, grid_on :bool):
        
        # Creating a new figure and axes
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111)
        
        # Adding a title at plot
        self.axes.set_title(title)
        
        # Adding x label at plot
        self.axes.set_xlabel(xlabel)
        
        # Adding y label at plot
        self.axes.set_ylabel(ylabel)
        
        # Adding grid at plot
        self.axes.grid(grid_on)
        
        # Adding x axes limits
        self.axes.set_xlim(xlim)
        
        # Adding y axes limit
        self.axes.set_ylim(ylim)
        
        # Turn on interactive mode
        plt.ion()
        
        
    def update(self, functions :list[tuple[Callable[...,None]]], step :float = 0.05):
        
        # Calling all the functions that need to re-draw in simulation
        for func,args in functions:
            func(*args)
            plt.draw()
        plt.pause(step)
        
    
    def move_rectangle(self,rectangle :patches.Rectangle, x :float, y :float):
        
        # x,y input is for the center,
        # so we need to calculate the left bottom coordinates
        
        width :float = rectangle.get_width()
        height :float = rectangle.get_height()
        
        # The x left bottom coordinate of the rectangle
        x_left_bottom :float = x - (width/2)
        
        # The y left bottom coordinate of the rectangle
        y_left_bottom :float = y - (height/2)
        
        # Setting the new position of rectangle
        rectangle.set_xy((x_left_bottom,y_left_bottom))
        
    def move_double_pendulum(self,q1 :float, q2 :float,base_coord :ndarray,l1 :float,l2 :float,lineDP, set_zero_offset :float = np.pi/2 ):# different dp from before cause the angles are not the same like epipedos braxionas
        
        x1 = l1 * np.cos(q1 + set_zero_offset)
        y1 = l1 * np.sin(q1 + set_zero_offset)
        x2 = l1 * np.cos(q1 + set_zero_offset) + l2 * np.cos(q2 + set_zero_offset)
        y2 = l1 * np.sin(q1 + set_zero_offset) + l2 * np.sin(q2 + set_zero_offset)
        
        lineDP.set_data([base_coord[0,0],x1 + base_coord[0,0],x2 + base_coord[0,0]],[base_coord[1,0],y1,y2])
        
    def _on_close(self,event):
        exit(1)
    
# Creating a sinusoidal control signal
def q(t,A = 1,f = 5,phi = np.pi/2):
    omega = 2*np.pi*f
    return A*np.sin(omega*t + phi)

def main():

    x = np.array([[0,np.pi,np.pi,0,0,0]]).T
    u = np.array([[0.5]]).T

    dp_c = DoublePendulumCart()
    # print(dp_c.dynamics(x,u))
    
    visual :Visualization = Visualization("Double_Pendulum_Cart","x","y",[-5,5],[-5,5],True)
    visual.fig.canvas.mpl_connect('close_event',visual._on_close)
    
    rect = patches.Rectangle((0.0 - 0.3, 0.0 - 0.15), 0.6, 0.3, linewidth=0.5, edgecolor='blue', facecolor='blue')
    visual.axes.add_patch(rect)
    
    # plt.ioff()
    # plt.show()
    
    
    lineDP, = visual.axes.plot([],[],"o-",markersize = 7,lw = 2)
    lineDP.set_data([],[])
    
    t = 0.0
    dt = 0.05
    
    states = []
    controls = []
    for i in range(100):
        
        u[0,0] = q(t,A = 8,f = 1)
        
        x = dp_c.runge_kutta4(x,u)
        states.append(x)
        print(u)
        controls.append(u[0,0])
        
        functions = [(visual.move_rectangle,(rect,x[0,0],0.0))
                    ,(visual.move_double_pendulum,(x[1,0],x[2,0],np.array([[x[0,0],0.0]]).T,dp_c.l2,dp_c.l3,lineDP))
                    ]
        
        visual.update(functions) 
        t+=dt
    
    plt.ioff()
    # plt.show()
    plt.pause(1)
    
    # Graphing the result of the position
    print("Plotting the results")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    K = 100
    
    # ax.plot([k * dt for k in range(K)], [x[0,0] for x in (states)], label="x")
    # ax.plot([k * dt for k in range(K)], [x[1,0] for x in (states)], label="θ1")
    # ax.plot([k * dt for k in range(K)], [x[2,0] for x in (states)], label="θ2")

    # ax.plot([k * dt for k in range(K)], [x[3,0] for x in (states)], label="$\dot{x}$")
    # ax.plot([k * dt for k in range(K)], [x[4,0] for x in (states)], label="$\dot{θ1}$")
    # ax.plot([k * dt for k in range(K)], [x[5,0] for x in (states)], label="$\dot{θ2}$")

    ax.plot([k * dt for k in range(K)], [u for u in controls], label="u")

    plt.xlabel("Time")

    plt.legend()
    plt.show()
    
    return 0

if __name__ == "__main__":
    main()
