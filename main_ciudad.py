"""Visor 3D de Gotham (Upper East Side + Coventry) con OpenGL.

Controles
    Arrastrar con el mouse (botón izquierdo) ... rotar / inclinar
    Flechas ................................... rotar / inclinar
    Rueda del mouse o W / S ................... acercar / alejar
    R ......................................... reiniciar cámara
    ESC ....................................... salir
"""

import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL import GL as gl
from OpenGL import GLU as glu

from Ciudad import Ciudad

ANCHO, ALTO = 1280, 800
FONDO = (14 / 255, 14 / 255, 20 / 255)

YAW0, PITCH0, DIST0 = 0.0, 30.0, 400.0
FOV = 48.0            # grados (campo de visión vertical)


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
    gl.glRotatef(pitch, 1.0, 0.0, 0.0)      # 90 = vista cenital
    gl.glRotatef(yaw, 0.0, 1.0, 0.0)
    gl.glScalef(1.0, 1.0, -1.0)             # el norte (z grande) queda al fondo


def main():
    pygame.init()
    pygame.display.set_mode((ANCHO, ALTO), DOUBLEBUF | OPENGL)
    reloj = pygame.time.Clock()

    configurar_gl()
    ciudad = Ciudad()
    info = ciudad.resumen()
    pygame.display.set_caption(
        f"Gotham City - {info['intersecciones']} intersecciones | "
        f"{info['tramos']} tramos | {info['edificios']} edificios | "
        f"{info['semaforos']} semáforos   "
        "(arrastrar: rotar | rueda / W S: zoom | R: reiniciar | ESC: salir)"
    )

    yaw, pitch, dist = YAW0, PITCH0, DIST0
    arrastrando = False

    corriendo = True
    while corriendo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    corriendo = False
                elif evento.key == pygame.K_r:
                    yaw, pitch, dist = YAW0, PITCH0, DIST0

            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                arrastrando = True

            elif evento.type == pygame.MOUSEBUTTONUP and evento.button == 1:
                arrastrando = False

            elif evento.type == pygame.MOUSEMOTION and arrastrando:
                yaw += evento.rel[0] * 0.4
                pitch += evento.rel[1] * 0.4

            elif evento.type == pygame.MOUSEWHEEL:
                dist *= 0.9 ** evento.y

        teclas = pygame.key.get_pressed()
        if teclas[pygame.K_LEFT]:
            yaw -= 1.5
        if teclas[pygame.K_RIGHT]:
            yaw += 1.5
        if teclas[pygame.K_UP]:
            pitch += 1.0
        if teclas[pygame.K_DOWN]:
            pitch -= 1.0
        if teclas[pygame.K_w]:
            dist *= 0.98
        if teclas[pygame.K_s]:
            dist *= 1.02

        pitch = max(5.0, min(90.0, pitch))
        dist = max(120.0, min(1500.0, dist))

        # Se redibuja cada cuadro: los semáforos cambian de color con el reloj
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        colocar_camara(yaw, pitch, dist)
        ciudad.dibujar()
        pygame.display.flip()

        reloj.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
