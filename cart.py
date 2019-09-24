import numpy as np 
import control
import pygame

g = -10
#TODO: (Low Priority) Make it so that user can grab and move the pendulum
class cart():

    def __init__(self, state, m=1., M=5., L=25.,controlled=False):
        self.state = state
        self.m = m
        self.M = M
        self.L = L
        self.d = 1.
        self.x = state[0]
        self.th = state[2]

        # Cart Data
        self.cw = 50 #cart width
        self.ch = 25 #cart height
        self.cx = self.x + (self.cw / 2)
        self.cy = self.ch / 2

        # Pendulum Data
        self.px = self.cx + 50 * np.sin(self.th)
        self.py = self.cy - 50 * np.cos(self.th)
        self.pr = 10
    
        self.dt = .16

        self.eigs = np.array([-1.1,-1.2,-1.3,-1.4])
        # self.K = getOptimalK(self.m,self.M,self.L,g,self.d) if controlled else np.zeros(4)
        self.K = calculateK(self.m,self.M,self.L,g,self.d,self.eigs) if controlled else np.zeros(4)

    def draw(self,screen):
        """Draw cart-pendulum on pygame screen"""
        w,h = screen.get_size()
        center = w/2
        pygame.draw.rect(screen, (255,0,0), (center + self.x, h/2 - self.ch, self.cw,self.ch))
        pygame.draw.line(screen, (255,0,0), (center + self.cx,h/2-self.cy),(center + self.px,h/2-self.py), 3 )
        pygame.draw.circle(screen, (255,0,0), (int(center + self.px),h/2 - int(self.py)), int(self.pr))

    def update(self):
        """Updates the internal state of the cart-pendulum system"""
        self.state = heuns(self.dt,self.state,self.m,self.M,self.L,g,self.d,
        -1*np.dot(self.K,(self.state - np.array([0,0,np.pi,0])))) 
        self.x = self.state[0]
        self.th = self.state[2]
        self.cx = self.x + (self.cw / 2)
        self.px = self.cx + 50 * np.sin(self.th)
        self.py = self.cy - 50 * np.cos(self.th)

#FIXME: Understand how LQR actually works and implement it
def getOptimalK(m,M,L,g,d):
        s = 1
        A = np.array([[0,1,0,0],[0,-d/M, -m*g/M,0],[0,0,0,1],[0,-s*d/(M*L), -s*(m+M)*g/(M*L),0]])
        B = [[0],[1/M],[0],[s/(M*L)]]
        Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,10,0],[0,0,0,100]])
        R = 0.0001
        K,S,E = control.lqr(A,B,Q,R)
        print(K)
        print(np.linalg.eig(A-B*K)[0])
        return K


def calculateK(m,M,L,g,d,eigs): 
    """calculate that places eigenvalues for A-B*K"""
     #TODO: Figure out how this works and implement it myself
    s = 1
    A = np.array([[0,1,0,0],[0,-d/M, -m*g/M,0],[0,0,0,1],[0,-s*d/(M*L), -s*(m+M)*g/(M*L),0]])
    B = [[0],[1/M],[0],[s/(M*L)]]

    K = control.place(A,B,eigs)
    
    return K

def heuns(dt, state,m,M,L,g,d,u):
    """Numerical ode solver using heun's method.
    returns updated state after one time step
    """
    ye = state + dt*stateDiff(state,m,M,L,g,d,u)
    return state + (dt/2)*(stateDiff(state,m,M,L,g,d,u) + stateDiff(ye,m,M,L,g,d,u))

def stateDiff(state,m,M,L,g,d,u):
    """ Calculate the differential for inverted pendulum on a cart given a state
    Returns as numpy array
    """
    dy = np.zeros(4,dtype=np.float64)
    Sy = np.sin(state[2])
    Cy = np.cos(state[2])
    D = m*L*L*(M+m*(1-Cy**2))

    dy[0] = state[1]
    dy[1] = (1/D)*(-(m**2)*(L**2)*g*Cy*Sy + m*(L**2)*(m*L*state[3]**2*Sy - d*state[1])) + u*m*L*L*(1/D)
    dy[2] = state[3]
    dy[3] = (1/D)*((m+M)*m*g*L*Sy - m*L*Cy*(m*L*(state[3]**2)*Sy - d*state[1])) - u*m*L*Cy*(1/D)

    return dy

if __name__ == "__main__":
    m, M, L, g, d = 1., 5., 2., -10., 1.
    K = getOptimalK(m,M,L,g,d)

    state = np.array([0,1,2,3])
    # print(np.outer(state,state) )
    # print("hello")