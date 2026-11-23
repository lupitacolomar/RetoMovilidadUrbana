import os
from OpenGL import GL as gl

from Ciudad import ESCALA_CARRO
from objloader import OBJ

COLORES_CARRO = [
    (0.64, 0.13, 0.11),  
    (0.15, 0.35, 0.78),   
    (0.92, 0.76, 0.10),  
    (0.18, 0.55, 0.30), 
    (0.93, 0.93, 0.93),   
    (0.48, 0.22, 0.62),   
]
MATERIAL_PINTURA = "Material.001"


Y_SUELO = 0.02 + 1.015 * ESCALA_CARRO


class _ModeloCarro(OBJ):
    generate_on_init = False    


class VistaCarros:
    def __init__(self):
        archivo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "carroceria.obj")
        self.modelos = []
        for color in COLORES_CARRO:
            modelo = _ModeloCarro(archivo)
            modelo.mtl[MATERIAL_PINTURA]["Kd"] = list(color)
            modelo.generate()
            self.modelos.append(modelo)

    def dibujar(self, carros):
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_NORMALIZE)     
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [150.0, 280.0, 100.0, 0.0])

        for carro in carros:
            gl.glPushMatrix()
            gl.glTranslatef(carro.x, Y_SUELO, carro.z)
            gl.glRotatef(carro.angulo, 0.0, 1.0, 0.0)
            gl.glScalef(ESCALA_CARRO, ESCALA_CARRO, ESCALA_CARRO)
            self.modelos[carro.color % len(self.modelos)].render()
            gl.glPopMatrix()

        gl.glDisable(gl.GL_NORMALIZE)
        gl.glDisable(gl.GL_LIGHTING)