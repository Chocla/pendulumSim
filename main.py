import pygame
import cart
import numpy as np
import control
(width, height ) = (600,300)
screen = pygame.display.set_mode((width,height))
pygame.display.set_caption('Pendulum Simulation')

c = cart.cart(np.array([0,0,np.pi + .1,0],dtype=np.float64),controlled=False)

pygame.display.flip()
clock = pygame.time.Clock()
running = True
fps = 60
while running:
    clock.tick(fps)
    c.update()
    c.draw(screen)
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((0,0,0))
