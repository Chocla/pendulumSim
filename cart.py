import numpy as np 
import pygame


class cart():

    def __init__(self, state, m=1, M=5, L=100):
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
        self.py = self.cy + self.L * np.cos(self.th)
        self.pr = 10*np.sqrt(self.m)
        pass

    def draw(self,screen):
        w,h = screen.get_size()
        center = w/2
        pygame.draw.rect(screen, (255,0,0), (center + self.x, h/2 - self.ch, self.cw,self.ch))
        pygame.draw.line(screen, (255,0,0), (center + self.cx,h/2-self.cy),(center + self.px,h/2-self.py), 3 )
        pygame.draw.circle(screen, (255,0,0), (center + int(self.px),h/2 - int(self.py)), int(self.pr))

    def update(self):
        self.state = eulers_method_cart(0.01,self.state,self.m,self.M,self.L,9.8,1,0)
        self.x = self.state[0]
        self.th = self.state[2]
        self.cx = self.x + (self.cw / 2)
        self.px = self.cx + self.L * np.sin(self.th)
        self.py = self.cy + self.L * np.cos(self.th)
        pass

def eulers_method_cart(dt, state,m,M,L,g,d,u):
    def pendulum_dy(state,m,M,L,g,d,u):
        dy = np.zeros(4,dtype=np.float64)
        Sy = np.sin(state[2])
        Cy = np.cos(state[2])
        D = m*L*L*(M+m*(1-Cy**2))

        dy[0] = state[1]
        dy[1] = (1/D)*(-(m**2)*(L**2)*g*Cy*Sy + m*(L**2)*(m*L*(state[3]**2)*Sy - d*state[1])) + m*L*L*(1/D)*u
        dy[2] = state[3]
        dy[3] = (1/D)*((m+M)*m*g*L*Sy - m*L*Cy*(m*L*state[3]**2*Sy - d*state[1])) - m*L*Cy*(1/D)*u
        return dy
    return state + dt*pendulum_dy(state,m,M,L,g,d,u)