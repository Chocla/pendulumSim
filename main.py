import pygame
import cart
import numpy as np

(width, height ) = (600,300)
screen = pygame.display.set_mode((width,height))

c = cart.cart([0,0.5,np.pi/2,2],m = 7, M = 15)

pygame.display.set_caption('Pendulum Simulation')


pygame.display.flip()
running = True
while running:
    c.update()
    c.draw(screen)
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((0,0,0))
