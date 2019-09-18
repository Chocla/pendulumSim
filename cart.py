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
        self.cw = 50*np.sqrt(M / 5) #cart width
        self.ch = 0.5 * self.cw #cart height
        self.cx = self.x + (self.cw / 2)
        self.cy = self.ch / 2
        self.px = self.cx + self.L * np.sin(self.th)
        self.py = self.cy + self.L * np.cos(self.th)
        self.pr = 15*np.sqrt(self.m)
        pass

    def draw(self,screen):
        w,h = screen.get_size()
        print(self.pr)
        pygame.draw.rect(screen, (255,0,0), (self.x, h - self.ch, self.cw,self.ch))
        pygame.draw.line(screen, (255,0,0), (self.cx,h-self.cy),(self.px,h-self.py), 3 )
        pygame.draw.circle(screen, (255,0,0), (int(self.px),h - int(self.py)), int(self.pr))
        # print(w,h
        pass

    def update(self):
        pass

