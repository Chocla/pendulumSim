import pygame
import cart
import numpy as np
import control
(width, height ) = (600,300)
screen = pygame.display.set_mode((width,height))

c = cart.cart(np.array([0,0,0,0],dtype=np.float64))

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
