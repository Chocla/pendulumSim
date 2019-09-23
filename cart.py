import numpy as np 
import control
import pygame

g = -10

class cart():

    def __init__(self, state, m=1., M=5., L=50.,controlled=False):
        self.state = state
        self.m = m
        self.M = M
        self.L = L
        self.x = state[0]
        self.th = state[2]
        self.cw = 45*np.sqrt(M / 5) #cart width
        self.ch = 0.5 * self.cw #cart height
        self.cx = self.x + (self.cw / 2)
        self.cy = self.ch / 2
        self.px = self.cx + self.L * np.sin(self.th)
        self.py = self.cy - self.L * np.cos(self.th)
        self.pr = 10*np.sqrt(self.m)
        self.dt = .16
        self.d = 1.
        self.eigs = np.array([-1.1,-1.2,-1.3,-1.4])
        # self.eigs = np.array([-2.1,-2.2,-2.3,-2.4])

        self.K = calculateK(self.m,self.M,self.L,g,self.d,self.eigs) if controlled else np.zeros(4)
        # print(self.K) #Expected: [-2.4,-8.75, 158.3,65.5]
        

    def draw(self,screen):
        w,h = screen.get_size()
        center = w/2
        # print(self.cx)
        pygame.draw.rect(screen, (255,0,0), (center + self.x, h/2 - self.ch, self.cw,self.ch))
        pygame.draw.line(screen, (255,0,0), (center + self.cx,h/2-self.cy),(center + self.px,h/2-self.py), 3 )
    
        pygame.draw.circle(screen, (255,0,0), (int(center + self.px),h/2 - int(self.py)), int(self.pr))

    
    def update(self):
        self.state = heuns_method_cart(self.dt,self.state,self.m,self.M,self.L,g,self.d,
        -1*np.dot(self.K,(self.state - np.array([0,0,np.pi,0])))) 
        # 0)
        self.x = self.state[0]
        self.th = self.state[2]
        self.cx = self.x + (self.cw / 2)
        self.px = self.cx + self.L * np.sin(self.th)
        self.py = self.cy - self.L * np.cos(self.th)
        
def calculateK(m,M,L,g,d,eigs): 
    s = 1
    A = np.array([[0,1,0,0],[0,-d/M, -m*g/M,0],[0,0,0,1],[0,-s*d/(M*L), -s*(m+M)*g/(M*L),0]])
    B = [[0],[1/M],[0],[s/(M*L)]]

    K = control.place(A,B,eigs) #TODO: Figure out how this works and implement it myself
    print(np.linalg.eig(A-np.outer(B,K)))
    return K

def heuns_method_cart(dt, state,m,M,L,g,d,u):
    ye = state + dt*pendulum_dy(state,m,M,L,g,d,u)
    return state + (dt/2)*(pendulum_dy(state,m,M,L,g,d,u) + pendulum_dy(ye,m,M,L,g,d,u))
# Shitty Solver, implement something better?? 
def eulers_method_cart(dt, state,m,M,L,g,d,u):
    return state + dt*pendulum_dy(state,m,M,L,g,d,u)
def pendulum_dy(state,m,M,L,g,d,u):
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
    K = calculateK(m,M,L,g,d,[-1.1,-1.2,-1.3,-1.4])
    print(K)
    print(np.dot(-K,[1,1,1,1]))
    state = np.array([0,1,2,3])
    # print(np.outer(state,state) )
    # print("hello")