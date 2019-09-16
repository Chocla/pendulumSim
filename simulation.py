import pygame
import numpy as np

(width, height ) = (600,300)
screen = pygame.display.set_mode((width,height))


class cart():
    def __init__(self,xPos, cW, cH, pL, pAngle, pMass, cMass):
        self.xPos = xPos
        self.cW = cW
        self.cH = cH
        self.cart = pygame.Rect(self.xPos,height - (self.cH + self.cW/4),self.cW,self.cH)
        self.pL = pL
        self.pAngle = pAngle
        self.pMass = pMass
        self.cmass = cMass 
    
    def display(self, screen, cartColor, ballColor):
        pygame.draw.rect(screen, cartColor, self.cart)
        pygame.draw.circle(screen,cartColor, (self.xPos+ self.cW/4, height - self.cW/8),self.cW/8)
        pygame.draw.circle(screen,cartColor, (self.xPos+ 3*self.cW/4, height - self.cW/8),self.cW/8)
        pygame.draw.line(screen,ballColor, (self.xPos + self.cW/2, height - (5*self.cW/8)), (self.xPos + self.cW/2 + self.pL*np.sin(self.pAngle), height - (5*self.cW/8) - self.pL*np.cos(self.pAngle)) )

myCart = cart(50,100,50,100,np.pi/4,0,0)
myCart.display(screen, (0,255,0),(255,0,0))


pygame.display.flip()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

