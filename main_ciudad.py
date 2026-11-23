"""
Controles
Arrastrar con el mouse (botón izquierdo) para inclinar la cámara
 W / S acercar / alejar
"""

import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL import GL as gl
from OpenGL import GLU as glu

from Ciudad import Ciudad
from Carro import crear_carros
from VistaCarro import VistaCarros

ANCHO, ALTO = 1280, 800
FONDO = (14 / 255, 14 / 255, 20 / 255)

YAW0, PITCH0, DIST0 = 0.0, 40.0, 250.0
FOV = 48.0          


def configurar_gl():
    gl.glClearColor(*FONDO, 1.0)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LEQUAL)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_LINE_SMOOTH)
    gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
    gl.glLineWidth(1.0)

    gl.glViewport(0, 0, ANCHO, ALTO)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(FOV, ANCHO / ALTO, 1.0, 4000.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)


def colocar_camara(yaw, pitch, dist):
    """Cámara orbital alrededor del centro de la ciudad (el origen)."""
    gl.glLoadIdentity()
    gl.glTranslatef(0.0, 0.0, -dist)
    gl.glRotatef(pitch, 1.0, 0.0, 0.0)   
    gl.glRotatef(yaw, 0.0, 1.0, 0.0)
    gl.glScalef(1.0, 1.0, -1.0)           


def main():
    pygame.init()
    pygame.display.set_mode((ANCHO, ALTO), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Gotham City - simulación de tránsito  "
                               "(mouse: rotar | W / S: zoom)")
    reloj = pygame.time.Clock()

    configurar_gl()
    ciudad = Ciudad()
    ciudad.auto_reloj = False             
    carros = crear_carros(ciudad)
    vista_carros = VistaCarros()

    yaw, pitch, dist = YAW0, PITCH0, DIST0
    arrastrando = False
    dt = 0.0

    corriendo = True
    while corriendo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False
            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                arrastrando = True
            elif evento.type == pygame.MOUSEBUTTONUP and evento.button == 1:
                arrastrando = False
            elif evento.type == pygame.MOUSEMOTION and arrastrando:
                yaw += evento.rel[0] * 0.4
                pitch += evento.rel[1] * 0.4

        teclas = pygame.key.get_pressed()
        if teclas[pygame.K_w]:
            dist *= 0.98
        if teclas[pygame.K_s]:
            dist *= 1.02

        pitch = max(5.0, min(90.0, pitch))
        dist = max(30.0, min(1500.0, dist))

    
        ciudad.actualizar(dt)
        for carro in carros:
            carro.actualizar(dt)

 
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        colocar_camara(yaw, pitch, dist)
        ciudad.dibujar()
        vista_carros.dibujar(carros)
        pygame.display.flip()

        dt = min(reloj.tick(60) / 1000.0, 0.05)

    pygame.quit()


if __name__ == "__main__":
    main()