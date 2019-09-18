import pygame
import cart
import numpy as np

(width, height ) = (600,300)
screen = pygame.display.set_mode((width,height))

c = cart.cart([0,0,np.pi/4,0])
print(c.x,300-c.ch, c.cw)

pygame.display.flip()
running = True
while running:
    c.draw(screen)
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
