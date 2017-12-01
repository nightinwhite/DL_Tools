from parallel_coordinate import *
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
import sys
import pygame
from pygame.locals import *
pygame.init()
tst_data = np.random.normal(0,1,[2000,5,3])
pc = Gen_parallel_coordinate(tst_data)
show_width = pc.img_width
show_height = pc.img_height
screen = pygame.display.set_mode([show_width, show_height])
bg_img ,draw_img = pc.gen_image()
while True:
    for e in pygame.event.get():
        if e.type == QUIT:
            sys.exit()
    screen.blit(bg_img,(0,0))
    screen.blit(draw_img,(0,0))
    pygame.display.update()
