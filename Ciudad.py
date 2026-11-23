"""Gotham City - zona Upper East Side + Coventry (Modelo 3D Avanzado).

Conserva al 100% la topología, planos, retícula y coordenadas originales:
    nodos   = 20 intersecciones (x, y, z)
    aristas = 31 tramos viales (autopista Schwartz Bypass, avenidas y calles)
              con sus atributos (tipo, límite de velocidad, carriles) para
              la simulación de agentes multiagente.
    semáforos = 62 semáforos sincronizados por eje (EW / NS) con ciclo de 2 fases.

Mejoras visuales implementadas (sin alterar los planos):
    1. Superficies viales sólidas con asfalto detallado y señalización horizontal
       (doble línea amarilla central en avenidas, líneas discontinuas blancas,
       pasos peatonales de cebra y líneas de detención en cada aproximación).
    2. Autopista Schwartz Bypass con camellón central sólido elevado y césped.
    3. Banquetas perimetrales elevadas en todas las manzanas con cordones de granito,
       pavimento y detalles de mobiliario urbano (hidrantes de bomberos).
    4. Edificios sólidos con iluminación volumétrica y ventanas procedimentales:
       - Wayne Tower: rascacielos Art Déco de 4 niveles, corona dorada y baliza roja.
       - Arkham Asylum: complejo gótico con 4 torres de vigilancia cónicas, óculo y reloj.
       - GCPD: cuartel policial con helipuerto ("H") y proyector de la Bat-Señal en las nubes.
       - Torres góticas, edificios Art Déco y almacenes industriales con chimeneas.
    5. Robinson Park con césped, senderos peatonales, bancas y árboles 3D volumétricos.
    6. Reservoir con espejo de agua profundo, muro de contención y embarcadero de madera.
    7. Semáforos 3D con postes, viseras parasol y lentes emisivas sincronizadas.
    8. Tres modos de visualización conmutables: Híbrido, Sólido y Alambre.
"""

import math
import random
import time
import numpy as np
from OpenGL import GL as gl

# ---------------------------------------------------------------------------
# Parámetros originales de la zona (Planos y Retícula)
# ---------------------------------------------------------------------------
COLUMNAS = 5          # avenidas (norte-sur)  -> 5 x 4 = 20 intersecciones
FILAS = 4             # calles (este-oeste)
SEP_AV = 62           # separación entre avenidas
SEP_CALLE = 46        # separación entre calles
ROTACION = -20        # grados: las avenidas se inclinan al oeste hacia el norte

FILA_PRINCIPAL = 1                         # fila de la avenida principal (recta)
NOMBRE_PRINCIPAL = "Schwartz Bypass"
CELDAS_VERDES = {(0, 2): "Reservoir",      # manzanas (col, fila) sin edificios
                 (1, 2): "Robinson Park"}
CELDA_ARKHAM = (0, 0)
CELDA_WAYNE = (1, 1)
CELDA_GCPD = (3, 1)                          # estación de policía con la Bat-Señal

MARGEN_X = 9.5        # separación entre el eje de la vía y los edificios
MARGEN_Z = 9.0
SEMILLA = 21          # misma ciudad en cada ejecución

# Semáforos: ciclo de dos fases (este-oeste / norte-sur)
T_VERDE = 8.0         # segundos en verde
T_AMARILLO = 2.0      # segundos en amarillo
ALTO_POSTE = 8.0      # altura del poste del semáforo
ALTO_CABEZA = 4.8     # caja con las 3 luces
RADIO_LUZ = 0.5       # radio de cada luz
ALTO_FAROLA = 10.0
N_ARBOLES = 16        # árboles en Robinson Park

# Dimensiones de carriles y vehículos
CARRO_LARGO_MODELO = 9.85
CARRO_ANCHO_MODELO = 3.92
ESCALA_CARRO = 0.45
CARRO_LARGO = CARRO_LARGO_MODELO * ESCALA_CARRO     # ~4.43
CARRO_ANCHO = CARRO_ANCHO_MODELO * ESCALA_CARRO     # ~1.76
FACTOR_CARRIL = 1.7
ANCHO_CARRIL = 3.0                                  # ~1.7 x 1.76 = 3.0
CAMELLON = 1.2                                      # separador central de la autopista
assert ANCHO_CARRIL >= 1.5 * CARRO_ANCHO, "el carro no cabe cómodo en el carril"

TIPOS = {
    "calle":     {"vel": 40, "carriles": 2, "camellon": 0.0},
    "avenida":   {"vel": 60, "carriles": 4, "camellon": 0.0},
    "autopista": {"vel": 80, "carriles": 4, "camellon": CAMELLON},
}
for _t in TIPOS.values():
    _t["carril"] = ANCHO_CARRIL
    _t["ancho"] = _t["carriles"] * ANCHO_CARRIL + _t["camellon"]

# Colores originales para modo alambre
COLORES = {
    "calle":      (105, 118, 140),
    "avenida":    (150, 178, 220),
    "autopista":  (235, 175, 70),
    "nodo":       (205, 205, 220),
    "zona":       (70, 150, 110),
    "caja":       (125, 135, 160),
    "escalonado": (160, 172, 205),
    "gotica":     (178, 150, 215),
    "almacen":    (105, 115, 128),
    "wayne":      (245, 215, 110),
    "arkham":     (225, 110, 110),
    "poste":      (190, 190, 205),
    "farola":     (255, 205, 110),
    "arbol":      (150, 120, 90),
    "senal":      (255, 238, 150),
    "gcpd":       (110, 170, 235),
}

COLOR_SEMAFORO = {
    "verde":    (60, 230, 110),
    "amarillo": (250, 220, 60),
    "rojo":     (240, 60, 60),
}
COLOR_APAGADO = {
    "verde":    (25, 85, 50),
    "amarillo": (95, 85, 25),
    "rojo":     (100, 32, 32),
}
ORDEN_LUCES = ("rojo", "amarillo", "verde")

MEZCLA = [("caja", 0.20), ("escalonado", 0.30), ("gotica", 0.38), ("almacen", 0.12)]


# ---------------------------------------------------------------------------
# Clases de la simulación
# ---------------------------------------------------------------------------
class Arista:
    """Tramo de calle entre dos intersecciones con sus propiedades de tráfico."""

    def __init__(self, a, b, tipo, nombre=None):
        self.a = a
        self.b = b
        self.tipo = tipo
        self.nombre = nombre
        self.ancho = TIPOS[tipo]["ancho"]
        self.limite_velocidad = TIPOS[tipo]["vel"]
        self.carriles = TIPOS[tipo]["carriles"]
        self.ancho_carril = TIPOS[tipo]["carril"]
        self.camellon = TIPOS[tipo]["camellon"]
        centros = [self.camellon / 2 + self.ancho_carril * (k + 0.5)
                   for k in range(self.carriles // 2)]
        self.carriles_ab = [-c for c in centros]
        self.carriles_ba = [c for c in centros]
        self.ruta = []          # puntos 3D del eje de la calle ya girados
        self.longitud = 0.0


class Semaforo:
    """Semáforo de 3 luces en un acceso de una intersección."""

    def __init__(self, nodo, eje, col, fila, x, z, ux, uz):
        self.nodo = nodo
        self.eje = eje
        self.col = col
        self.fila = fila
        self.x, self.z = x, z
        self.ux, self.uz = ux, uz
        self.estado = "rojo"


class Edificio:
    def __init__(self, tipo, altura, huella2d, nombre=None):
        self.tipo = tipo
        self.altura = altura
        self.nombre = nombre
        self.huella2d = huella2d    # 4 esquinas (x, z) en retícula
        self.huella = []            # 4 esquinas 3D ya giradas


# ---------------------------------------------------------------------------
# Generador de Geometría 3D Sólida y Alambre
# ---------------------------------------------------------------------------
class _GeometriaBuilder:
    def __init__(self, cx, cz, rot_deg):
        self.cx = cx
        self.cz = cz
        ang = math.radians(rot_deg)
        self.cos_a = math.cos(ang)
        self.sin_a = math.sin(ang)

        self.solidos_v = []
        self.lineas_v = []
        self.transp_v = []

    def girar_pto(self, x, y, z):
        dx = x - self.cx
        dz = z - self.cz
        rx = self.cos_a * dx + self.sin_a * dz
        rz = -self.sin_a * dx + self.cos_a * dz
        return (rx, y, rz)

    def girar_vec(self, vx, vy, vz):
        rx = self.cos_a * vx + self.sin_a * vz
        rz = -self.sin_a * vx + self.cos_a * vz
        return (rx, vy, rz)

    def add_tri(self, p0, p1, p2, color, normal=None, transp=False, rotar=True):
        gp0 = self.girar_pto(*p0) if rotar else p0
        gp1 = self.girar_pto(*p1) if rotar else p1
        gp2 = self.girar_pto(*p2) if rotar else p2

        if normal is None:
            ax, ay, az = gp1[0] - gp0[0], gp1[1] - gp0[1], gp1[2] - gp0[2]
            bx, by, bz = gp2[0] - gp0[0], gp2[1] - gp0[1], gp2[2] - gp0[2]
            nx = ay * bz - az * by
            ny = az * bx - ax * bz
            nz = ax * by - ay * bx
            mag = math.hypot(nx, ny, nz) or 1.0
            normal = (nx / mag, ny / mag, nz / mag)
        elif rotar:
            normal = self.girar_vec(*normal)

        c = [comp / 255.0 if comp > 1.0 else float(comp) for comp in color]
        if len(c) == 3:
            c.append(1.0)
        r, g, b, a = c

        target = self.transp_v if transp else self.solidos_v
        for p in (gp0, gp1, gp2):
            target.extend([p[0], p[1], p[2], normal[0], normal[1], normal[2], r, g, b, a])

    def add_quad(self, p0, p1, p2, p3, color, normal=None, transp=False, rotar=True):
        self.add_tri(p0, p1, p2, color, normal, transp, rotar)
        self.add_tri(p0, p2, p3, color, normal, transp, rotar)

    def add_linea(self, p0, p1, color, rotar=True):
        gp0 = self.girar_pto(*p0) if rotar else p0
        gp1 = self.girar_pto(*p1) if rotar else p1
        c = [comp / 255.0 if comp > 1.0 else float(comp) for comp in color]
        r, g, b = c[:3]
        self.lineas_v.extend([gp0[0], gp0[1], gp0[2], r, g, b])
        self.lineas_v.extend([gp1[0], gp1[1], gp1[2], r, g, b])

    def add_caja(self, x0, y0, z0, x1, y1, z1, col_pared, col_techo=None, rotar=True, con_borde=True):
        if col_techo is None:
            col_techo = col_pared
        self.add_quad((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), col_techo, (0, 1, 0), rotar=rotar)
        self.add_quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), col_pared, (0, 0, 1), rotar=rotar)
        self.add_quad((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), col_pared, (0, 0, -1), rotar=rotar)
        self.add_quad((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), col_pared, (1, 0, 0), rotar=rotar)
        self.add_quad((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), col_pared, (-1, 0, 0), rotar=rotar)

        if con_borde:
            bc = (col_pared[0] * 0.45, col_pared[1] * 0.45, col_pared[2] * 0.45)
            for ex, ez in ((x0, z0), (x1, z0), (x1, z1), (x0, z1)):
                self.add_linea((ex, y0, ez), (ex, y1, ez), bc, rotar=rotar)
            top_esq = [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
            for k in range(4):
                self.add_linea(top_esq[k], top_esq[(k + 1) % 4], bc, rotar=rotar)

    def add_piramide(self, x0, z0, x1, z1, y_base, alto, color, rotar=True):
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        punta = (cx, y_base + alto, cz)
        b = [(x0, y_base, z0), (x1, y_base, z0), (x1, y_base, z1), (x0, y_base, z1)]
        for i in range(4):
            self.add_tri(b[i], b[(i + 1) % 4], punta, color, rotar=rotar)
        bc = (color[0] * 0.55, color[1] * 0.55, color[2] * 0.55)
        for i in range(4):
            self.add_linea(b[i], punta, bc, rotar=rotar)
            self.add_linea(b[i], b[(i + 1) % 4], bc, rotar=rotar)

    def add_gable(self, x0, z0, x1, z1, y_base, alto, color, rotar=True):
        zc, xc = (z0 + z1) / 2, (x0 + x1) / 2
        if (x1 - x0) >= (z1 - z0):
            c0, c1 = (x0, y_base + alto, zc), (x1, y_base + alto, zc)
            self.add_quad((x0, y_base, z0), (x1, y_base, z0), c1, c0, color, rotar=rotar)
            self.add_quad(c0, c1, (x1, y_base, z1), (x0, y_base, z1), color, rotar=rotar)
            self.add_tri((x0, y_base, z1), (x0, y_base, z0), c0, color, rotar=rotar)
            self.add_tri((x1, y_base, z0), (x1, y_base, z1), c1, color, rotar=rotar)
        else:
            c0, c1 = (xc, y_base + alto, z0), (xc, y_base + alto, z1)
            self.add_quad((x0, y_base, z0), c0, c1, (x0, y_base, z1), color, rotar=rotar)
            self.add_quad(c0, (x1, y_base, z0), (x1, y_base, z1), c1, color, rotar=rotar)
            self.add_tri((x0, y_base, z0), (x1, y_base, z0), c0, color, rotar=rotar)
            self.add_tri((x1, y_base, z1), (x0, y_base, z1), c1, color, rotar=rotar)


# ---------------------------------------------------------------------------
# Clase Ciudad Principal
# ---------------------------------------------------------------------------
class Ciudad:
    def __init__(self):
        self.idx = {}
        self.nodos = []
        self.aristas = []
        self.edificios = []
        self.zonas = {}
        self.semaforos = []
        self.n_farolas = 0
        self.n_arboles = 0
        self.tiempo = 0.0
        self.auto_reloj = True
        self._t_ultimo = None
        self.rng = random.Random(SEMILLA)

        # Modos de visualización: "hibrido" (sólido + aristas), "solido", "alambre"
        self.modo = "hibrido"
        self._display_list = None
        self.luces_semaforos_dinamicas = []

        # Arreglos de alambre original para compatibilidad total
        self.vertices = None
        self.colores_v = None

        self._construir()

    @staticmethod
    def _posicion_base(c, r):
        return (c * SEP_AV, (FILAS - 1 - r) * SEP_CALLE)

    @staticmethod
    def _rect_celda(c, r):
        x0, x1 = c * SEP_AV, (c + 1) * SEP_AV
        z0 = (FILAS - 1 - (r + 1)) * SEP_CALLE
        z1 = (FILAS - 1 - r) * SEP_CALLE
        return x0, z0, x1, z1

    def _construir(self):
        pos2d = []
        aristas = []

        # Nodos
        for r in range(FILAS):
            for c in range(COLUMNAS):
                self.idx[(c, r)] = len(pos2d)
                pos2d.append(self._posicion_base(c, r))

        # Aristas
        for r in range(FILAS):
            for c in range(COLUMNAS):
                a = self.idx[(c, r)]
                if c + 1 < COLUMNAS:
                    if r == FILA_PRINCIPAL:
                        aristas.append(Arista(a, self.idx[(c + 1, r)], "autopista", NOMBRE_PRINCIPAL))
                    else:
                        aristas.append(Arista(a, self.idx[(c + 1, r)], "calle"))
                if r + 1 < FILAS:
                    aristas.append(Arista(a, self.idx[(c, r + 1)], "avenida"))

        rutas2d = [[pos2d[ar.a], pos2d[ar.b]] for ar in aristas]

        radio = [0.0] * len(pos2d)
        for ar in aristas:
            radio[ar.a] = max(radio[ar.a], ar.ancho / 2 + 0.7)
            radio[ar.b] = max(radio[ar.b], ar.ancho / 2 + 0.7)

        xs = [p[0] for p in pos2d]
        zs = [p[1] for p in pos2d]
        self._cx, self._cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2

        bld = _GeometriaBuilder(self._cx, self._cz, ROTACION)
        self.bld = bld

        # 1. Base del terreno
        min_x, max_x = min(xs) - 70, max(xs) + 70
        min_z, max_z = min(zs) - 70, max(zs) + 70
        bld.add_quad((min_x, -0.05, min_z), (min_x, -0.05, max_z),
                     (max_x, -0.05, max_z), (max_x, -0.05, min_z),
                     (16, 18, 24), (0, 1, 0), rotar=True)

        # 2. Banquetas perimetrales elevadas
        col_bordillo = (125, 130, 140)
        col_banqueta = (80, 84, 94)
        col_junta = (65, 68, 76)

        for r in range(FILAS - 1):
            for c in range(COLUMNAS - 1):
                if (c, r) in CELDAS_VERDES:
                    continue
                x0, z0, x1, z1 = self._rect_celda(c, r)
                bx0 = x0 + MARGEN_X - 1.2
                bx1 = x1 - MARGEN_X + 1.2
                bz0 = z0 + MARGEN_Z - 1.2
                bz1 = z1 - MARGEN_Z + 1.2
                bld.add_caja(bx0, 0.0, bz0, bx1, 0.22, bz1, col_bordillo, col_banqueta, rotar=True, con_borde=True)
                for gx in np.linspace(bx0 + 4, bx1 - 4, int((bx1 - bx0)/8)):
                    bld.add_linea((gx, 0.225, bz0), (gx, 0.225, bz1), col_junta, rotar=True)
                for gz in np.linspace(bz0 + 4, bz1 - 4, int((bz1 - bz0)/8)):
                    bld.add_linea((bx0, 0.225, gz), (bx1, 0.225, gz), col_junta, rotar=True)

                # Hidrantes de bomberos
                hx, hz = bx1 - 1.8, bz0 + 1.8
                bld.add_caja(hx - 0.25, 0.22, hz - 0.25, hx + 0.25, 0.95, hz + 0.25,
                             (210, 35, 35), (230, 45, 45), rotar=True)
                bld.add_caja(hx - 0.4, 0.65, hz - 0.12, hx + 0.4, 0.80, hz + 0.12,
                             (240, 190, 40), rotar=True)

        # 3. Calles y Asfalto
        col_asfalto = (32, 34, 40)
        col_asfalto_hwy = (24, 26, 32)
        col_linea_blanca = (240, 240, 245)
        col_linea_amarilla = (250, 195, 30)

        for ar, ruta in zip(aristas, rutas2d):
            Q = self._recortar(ruta, radio[ar.a], radio[ar.b])
            (x_a, z_a), (x_b, z_b) = Q[0], Q[1]
            dx, dz = x_b - x_a, z_b - z_a
            L = math.hypot(dx, dz) or 1.0
            ux, uz = dx / L, dz / L
            nx, nz = -uz, ux
            hw = ar.ancho / 2.0

            c_asf = col_asfalto_hwy if ar.tipo == "autopista" else col_asfalto
            p0 = (x_a - nx * hw, 0.02, z_a - nz * hw)
            p1 = (x_b - nx * hw, 0.02, z_b - nz * hw)
            p2 = (x_b + nx * hw, 0.02, z_b + nz * hw)
            p3 = (x_a + nx * hw, 0.02, z_a + nz * hw)
            bld.add_quad(p0, p1, p2, p3, c_asf, (0, 1, 0), rotar=True)
            bld.add_linea((p0[0], 0.03, p0[2]), (p1[0], 0.03, p1[2]), (55, 60, 70), rotar=True)
            bld.add_linea((p3[0], 0.03, p3[2]), (p2[0], 0.03, p2[2]), (55, 60, 70), rotar=True)

            if ar.tipo == "calle":
                n_tramos = int(L / 3.8)
                for k in range(n_tramos):
                    t0 = (k + 0.25) / n_tramos
                    t1 = (k + 0.75) / n_tramos
                    la = (x_a + dx * t0, 0.035, z_a + dz * t0)
                    lb = (x_a + dx * t1, 0.035, z_a + dz * t1)
                    bld.add_quad((la[0] - nx * 0.12, 0.035, la[2] - nz * 0.12),
                                 (lb[0] - nx * 0.12, 0.035, lb[2] - nz * 0.12),
                                 (lb[0] + nx * 0.12, 0.035, lb[2] + nz * 0.12),
                                 (la[0] + nx * 0.12, 0.035, la[2] + nz * 0.12),
                                 col_linea_blanca, (0, 1, 0), rotar=True)

            elif ar.tipo == "avenida":
                for off in (-0.22, 0.22):
                    la = (x_a + nx * off, 0.035, z_a + nz * off)
                    lb = (x_b + nx * off, 0.035, z_b + nz * off)
                    bld.add_quad((la[0] - nx * 0.09, 0.035, la[2] - nz * 0.09),
                                 (lb[0] - nx * 0.09, 0.035, lb[2] - nz * 0.09),
                                 (lb[0] + nx * 0.09, 0.035, lb[2] + nz * 0.09),
                                 (la[0] + nx * 0.09, 0.035, la[2] + nz * 0.09),
                                 col_linea_amarilla, (0, 1, 0), rotar=True)
                for sign in (-1, 1):
                    off = sign * ANCHO_CARRIL
                    n_tramos = int(L / 3.8)
                    for k in range(n_tramos):
                        t0 = (k + 0.25) / n_tramos
                        t1 = (k + 0.75) / n_tramos
                        la = (x_a + dx * t0 + nx * off, 0.035, z_a + dz * t0 + nz * off)
                        lb = (x_a + dx * t1 + nx * off, 0.035, z_a + dz * t1 + nz * off)
                        bld.add_quad((la[0] - nx * 0.1, 0.035, la[2] - nz * 0.1),
                                     (lb[0] - nx * 0.1, 0.035, lb[2] - nz * 0.1),
                                     (lb[0] + nx * 0.1, 0.035, lb[2] + nz * 0.1),
                                     (la[0] + nx * 0.1, 0.035, la[2] + nz * 0.1),
                                     col_linea_blanca, (0, 1, 0), rotar=True)

            elif ar.tipo == "autopista":
                hw_c = CAMELLON / 2.0
                c0 = (x_a - nx * hw_c, 0.0, z_a - nz * hw_c)
                c1 = (x_b - nx * hw_c, 0.0, z_b - nz * hw_c)
                c2 = (x_b + nx * hw_c, 0.0, z_b + nz * hw_c)
                c3 = (x_a + nx * hw_c, 0.0, z_a + nz * hw_c)
                bld.add_quad((c0[0], 0.35, c0[2]), (c1[0], 0.35, c1[2]),
                             (c2[0], 0.35, c2[2]), (c3[0], 0.35, c3[2]),
                             (48, 72, 44), (0, 1, 0), rotar=True)
                bld.add_quad((c0[0], 0.0, c0[2]), (c1[0], 0.0, c1[2]),
                             (c1[0], 0.35, c1[2]), (c0[0], 0.35, c0[2]),
                             (160, 165, 175), (-nx, 0, -nz), rotar=True)
                bld.add_quad((c3[0], 0.35, c3[2]), (c2[0], 0.35, c2[2]),
                             (c2[0], 0.0, c2[2]), (c3[0], 0.0, c3[2]),
                             (160, 165, 175), (nx, 0, nz), rotar=True)
                for sign in (-1, 1):
                    off = sign * (hw_c + 0.3)
                    la = (x_a + nx * off, 0.035, z_a + nz * off)
                    lb = (x_b + nx * off, 0.035, z_b + nz * off)
                    bld.add_linea(la, lb, col_linea_blanca, rotar=True)
                    off_ext = sign * (hw_c + ANCHO_CARRIL)
                    n_tramos = int(L / 3.8)
                    for k in range(n_tramos):
                        t0 = (k + 0.25) / n_tramos
                        t1 = (k + 0.75) / n_tramos
                        sla = (x_a + dx * t0 + nx * off_ext, 0.035, z_a + dz * t0 + nz * off_ext)
                        slb = (x_a + dx * t1 + nx * off_ext, 0.035, z_a + dz * t1 + nz * off_ext)
                        bld.add_linea(sla, slb, col_linea_blanca, rotar=True)

            for p_end, dir_s in ((Q[0], 1.0), (Q[1], -1.0)):
                p_stop = (p_end[0] + ux * dir_s * 1.5, 0.038, p_end[1] + uz * dir_s * 1.5)
                bld.add_quad((p_stop[0] - nx * (hw - 0.5), 0.038, p_stop[2] - nz * (hw - 0.5)),
                             (p_stop[0] + nx * (hw - 0.5), 0.038, p_stop[2] + nz * (hw - 0.5)),
                             (p_stop[0] + nx * (hw - 0.5) + ux * dir_s * 0.4, 0.038, p_stop[2] + nz * (hw - 0.5) + uz * dir_s * 0.4),
                             (p_stop[0] - nx * (hw - 0.5) + ux * dir_s * 0.4, 0.038, p_stop[2] - nz * (hw - 0.5) + uz * dir_s * 0.4),
                             col_linea_blanca, (0, 1, 0), rotar=True)
                n_franjas = int((hw * 2 - 1.0) / 0.8)
                p_cebra = (p_end[0] + ux * dir_s * 0.4, 0.038, p_end[1] + uz * dir_s * 0.4)
                for f in range(n_franjas):
                    f_off = -hw + 0.8 + f * 0.8
                    ca = (p_cebra[0] + nx * (f_off - 0.25), 0.038, p_cebra[2] + nz * (f_off - 0.25))
                    cb = (p_cebra[0] + nx * (f_off + 0.25), 0.038, p_cebra[2] + nz * (f_off + 0.25))
                    bld.add_quad(ca, cb,
                                 (cb[0] + ux * dir_s * 0.9, 0.038, cb[2] + uz * dir_s * 0.9),
                                 (ca[0] + ux * dir_s * 0.9, 0.038, ca[2] + uz * dir_s * 0.9),
                                 col_linea_blanca, (0, 1, 0), rotar=True)

        # 4. Cruces de intersecciones
        for i, (ix, iz) in enumerate(pos2d):
            h = radio[i]
            esq = [(ix - h, 0.022, iz - h), (ix + h, 0.022, iz - h),
                   (ix + h, 0.022, iz + h), (ix - h, 0.022, iz + h)]
            bld.add_quad(esq[0], esq[1], esq[2], esq[3], (26, 28, 34), (0, 1, 0), rotar=True)
            for k in range(4):
                bld.add_linea(esq[k], esq[(k + 1) % 4], (50, 55, 65), rotar=True)

        # 5. Zonas verdes
        zonas2d = {}
        for (c, r), nombre in CELDAS_VERDES.items():
            zonas2d[nombre] = self._elipse_celda(c, r)

        res_pts = zonas2d.get("Reservoir")
        if res_pts:
            cx_res = sum(p[0] for p in res_pts) / len(res_pts)
            cz_res = sum(p[1] for p in res_pts) / len(res_pts)
            n_p = len(res_pts)
            for k in range(n_p):
                p0, p1 = res_pts[k], res_pts[(k + 1) % n_p]
                bld.add_quad((p0[0], 0.0, p0[1]), (p1[0], 0.0, p1[1]),
                             (p1[0], 0.45, p1[1]), (p0[0], 0.45, p0[1]),
                             (85, 90, 100), rotar=True)
                bld.add_linea((p0[0], 0.45, p0[1]), (p1[0], 0.45, p1[1]), (130, 135, 145), rotar=True)
            for k in range(n_p):
                p0, p1 = res_pts[k], res_pts[(k + 1) % n_p]
                bld.add_tri((cx_res, 0.20, cz_res), (p0[0], 0.20, p0[1]), (p1[0], 0.20, p1[1]),
                            (22, 75, 95), (0, 1, 0), rotar=True)
            px0, pz0 = cx_res + 9.5, cz_res
            bld.add_caja(px0 - 3.2, 0.20, pz0 - 1.2, px0 + 3.2, 0.30, pz0 + 1.2, (100, 70, 45), (120, 85, 55), rotar=True)
            for m_off in (-2.6, 0.0, 2.6):
                bld.add_caja(px0 + m_off - 0.2, 0.20, pz0 - 1.4, px0 + m_off + 0.2, 0.60, pz0 - 1.0, (65, 45, 30), rotar=True)
                bld.add_caja(px0 + m_off - 0.2, 0.20, pz0 + 1.0, px0 + m_off + 0.2, 0.60, pz0 + 1.4, (65, 45, 30), rotar=True)

        park_pts = zonas2d.get("Robinson Park")
        if park_pts:
            cx_pk = sum(p[0] for p in park_pts) / len(park_pts)
            cz_pk = sum(p[1] for p in park_pts) / len(park_pts)
            n_p = len(park_pts)
            for k in range(n_p):
                p0, p1 = park_pts[k], park_pts[(k + 1) % n_p]
                bld.add_quad((p0[0], 0.0, p0[1]), (p1[0], 0.0, p1[1]),
                             (p1[0], 0.25, p1[1]), (p0[0], 0.25, p0[1]),
                             (80, 85, 80), rotar=True)
                bld.add_tri((cx_pk, 0.25, cz_pk), (p0[0], 0.25, p0[1]), (p1[0], 0.25, p1[1]),
                            (30, 56, 36), (0, 1, 0), rotar=True)

            rx_p = max(p[0] for p in park_pts) - cx_pk
            rz_p = max(p[1] for p in park_pts) - cz_pk
            for k in range(n_p):
                t0 = 2 * math.pi * k / n_p
                t1 = 2 * math.pi * (k + 1) / n_p
                a0 = (cx_pk + rx_p * 0.62 * math.cos(t0), 0.26, cz_pk + rz_p * 0.62 * math.sin(t0))
                a1 = (cx_pk + rx_p * 0.62 * math.cos(t1), 0.26, cz_pk + rz_p * 0.62 * math.sin(t1))
                bld.add_linea(a0, a1, (115, 120, 110), rotar=True)

            rng_arb = random.Random(SEMILLA + 7)
            for _ in range(N_ARBOLES):
                while True:
                    u, v = rng_arb.uniform(-1, 1), rng_arb.uniform(-1, 1)
                    if 0.06 < u * u + v * v <= 0.74:
                        break
                tx, tz = cx_pk + u * rx_p, cz_pk + v * rz_p
                h_tronco = rng_arb.uniform(6.5, 10.5)
                bld.add_caja(tx - 0.4, 0.25, tz - 0.4, tx + 0.4, 0.25 + h_tronco, tz + 0.4,
                             (65, 45, 30), (55, 38, 26), rotar=True)
                n_copas = rng_arb.choice([2, 3])
                for cp in range(n_copas):
                    y_cp = 0.25 + h_tronco * (0.42 + cp * 0.26)
                    r_cp = rng_arb.uniform(2.6, 4.2) * (1.0 - cp * 0.22)
                    alto_cp = rng_arb.uniform(3.8, 5.2)
                    col_fol = rng_arb.choice([(24, 48, 28), (30, 56, 34), (18, 40, 24)])
                    bld.add_piramide(tx - r_cp, tz - r_cp, tx + r_cp, tz + r_cp, y_cp, alto_cp, col_fol, rotar=True)
                self.n_arboles += 1

                if rng_arb.random() < 0.4:
                    bx, bz = tx + 2.0, tz + 1.0
                    bld.add_caja(bx - 1.2, 0.25, bz - 0.4, bx + 1.2, 0.65, bz + 0.4,
                                 (90, 60, 38), (110, 75, 45), rotar=True)

        # 6. Farolas y Semáforos
        for ar, ruta in zip(aristas, rutas2d):
            (ax, az), (bx, bz) = ruta[0], ruta[-1]
            L = math.hypot(bx - ax, bz - az)
            dx, dz = (bx - ax) / L, (bz - az) / L
            nx, nz = -dz, dx
            for t_frac, lado in ((1 / 3, +1), (2 / 3, -1)):
                px, pz = ax + (bx - ax) * t_frac, az + (bz - az) * t_frac
                fx = px + lado * nx * (ar.ancho / 2 + 0.9)
                fz = pz + lado * nz * (ar.ancho / 2 + 0.9)
                bld.add_caja(fx - 0.2, 0.0, fz - 0.2, fx + 0.2, ALTO_FAROLA, fz + 0.2,
                             (50, 55, 62), (65, 70, 78), rotar=True)
                fxa = fx - lado * nx * 2.2
                fza = fz - lado * nz * 2.2
                bld.add_linea((fx, ALTO_FAROLA, fz), (fxa, ALTO_FAROLA, fza), (75, 80, 90), rotar=True)
                bld.add_caja(fxa - 0.35, ALTO_FAROLA - 0.8, fza - 0.35, fxa + 0.35, ALTO_FAROLA, fza + 0.35,
                             (255, 230, 140), (255, 245, 180), rotar=True)
                self.n_farolas += 1

        for ar, ruta in zip(aristas, rutas2d):
            (ax_, az_), (bx_, bz_) = ruta[0], ruta[-1]
            L = math.hypot(bx_ - ax_, bz_ - az_)
            dx, dz = (bx_ - ax_) / L, (bz_ - az_) / L
            eje = "EW" if abs(dx) > abs(dz) else "NS"

            for nodo, (px, pz), (ux, uz) in ((ar.a, (ax_, az_), (dx, dz)),
                                             (ar.b, (bx_, bz_), (-dx, -dz))):
                rx_, rz_ = -uz, ux
                sx = px + ux * (radio[nodo] + 1.2) + rx_ * (ar.ancho / 2 + 1.0)
                sz = pz + uz * (radio[nodo] + 1.2) + rz_ * (ar.ancho / 2 + 1.0)
                bld.add_caja(sx - 0.25, 0.0, sz - 0.25, sx + 0.25, ALTO_POSTE + ALTO_CABEZA, sz + 0.25,
                             (42, 45, 50), (55, 60, 65), rotar=True)
                arm_x = sx - rx_ * 1.8
                arm_z = sz - rz_ * 1.8
                bld.add_linea((sx, ALTO_POSTE + ALTO_CABEZA - 0.5, sz),
                              (arm_x, ALTO_POSTE + ALTO_CABEZA - 0.5, arm_z), (60, 65, 72), rotar=True)

                y0_c, y1_c = ALTO_POSTE, ALTO_POSTE + ALTO_CABEZA
                cx0, cz0 = sx + ux * 0.3, sz + uz * 0.3
                bld.add_caja(cx0 - 0.55, y0_c, cz0 - 0.55, cx0 + 0.55, y1_c, cz0 + 0.55,
                             (26, 28, 34), (18, 20, 24), rotar=True)

                col = next(c for (c, f), i in self.idx.items() if i == nodo)
                fila = next(f for (c, f), i in self.idx.items() if i == nodo)
                sem = Semaforo(nodo, eje, col, fila, sx, sz, ux, uz)
                self.semaforos.append(sem)

                luces_sem = []
                for f_y, nombre_luz in zip((0.8, 0.5, 0.2), ORDEN_LUCES):
                    ly = y0_c + ALTO_CABEZA * f_y
                    lx_face = cx0 + ux * 0.62
                    lz_face = cz0 + uz * 0.62
                    gp = bld.girar_pto(lx_face, ly, lz_face)
                    gnorm = bld.girar_vec(ux, 0.0, uz)
                    luces_sem.append((gp, gnorm, nombre_luz))
                self.luces_semaforos_dinamicas.append((sem, luces_sem))

        # 7. Edificios
        ejes = [(pos2d[ar.a], pos2d[ar.b], ar.ancho / 2) for ar in aristas]
        edificios = self._generar_edificios_solidos(ejes, bld)
        self.edificios = edificios

        def a3d(lista2d):
            return [bld.girar_pto(x, 0.0, z) for x, z in lista2d]

        self.nodos = a3d(pos2d)
        for ar, ruta in zip(aristas, rutas2d):
            ar.ruta = a3d(ruta)
            ar.longitud = sum(math.dist(p, q) for p, q in zip(ar.ruta[:-1], ar.ruta[1:]))
        self.aristas = aristas

        for nombre, poligono in zonas2d.items():
            self.zonas[nombre] = a3d(poligono)

        for e in edificios:
            e.huella = a3d(e.huella2d)

        self.arr_solidos = np.array(bld.solidos_v, dtype=np.float32)
        self.arr_lineas = np.array(bld.lineas_v, dtype=np.float32)
        self.arr_transp = np.array(bld.transp_v, dtype=np.float32) if bld.transp_v else None

        # Buffers de alambre para modo compatibilidad
        self.vertices = self.arr_lineas[0::2]
        self.colores_v = np.repeat(np.array([[120, 140, 180]], dtype=np.float32) / 255.0, len(self.vertices), axis=0)

        self._display_list = None
        self.actualizar(0.0)

    def _generar_edificios_solidos(self, ejes, bld):
        rng = self.rng
        edificios = []
        centro_x = (COLUMNAS - 1) * SEP_AV / 2
        centro_z = (FILAS - 1) * SEP_CALLE / 2

        colores_fachada = [
            (110, 115, 125),
            (138, 128, 118),
            (90, 100, 115),
            (105, 60, 50),
            (55, 70, 85),
        ]
        colores_techo = [
            (42, 45, 52),
            (52, 56, 64),
            (38, 40, 46),
        ]

        for r in range(FILAS - 1):
            for c in range(COLUMNAS - 1):
                if (c, r) in CELDAS_VERDES:
                    continue
                x0, z0, x1, z1 = self._rect_celda(c, r)
                cx, cz = (x0 + x1) / 2, (z0 + z1) / 2

                if (c, r) == CELDA_ARKHAM:
                    edificios.append(self._arkham_solido(cx, cz, bld))
                    continue
                if (c, r) == CELDA_WAYNE:
                    edificios.append(self._wayne_solido(cx, cz, bld))
                    continue
                if (c, r) == CELDA_GCPD:
                    edificios.append(self._gcpd_solido(cx, cz, bld))
                    continue

                ix0, ix1 = x0 + MARGEN_X, x1 - MARGEN_X
                iz0, iz1 = z0 + MARGEN_Z, z1 - MARGEN_Z
                lw, ld = (ix1 - ix0) / 2, (iz1 - iz0) / 2

                lotes = []
                for i in range(2):
                    for j in range(2):
                        if rng.random() < 0.08:
                            continue
                        lotes.append((
                            ix0 + lw * (i + 0.5) + rng.uniform(-1.2, 1.2),
                            iz0 + ld * (j + 0.5) + rng.uniform(-1.0, 1.0),
                            (lw - 3.5) * rng.uniform(0.75, 1.0),
                            (ld - 3.5) * rng.uniform(0.75, 1.0)
                        ))

                for lcx, lcz, w, d in lotes:
                    huella = self._huella_libre(lcx, lcz, w, d, ejes)
                    if huella is None:
                        continue
                    dist = math.hypot((lcx - centro_x) / (centro_x or 1),
                                      (lcz - centro_z) / (centro_z or 1))
                    altura = rng.uniform(24, 50) + 45 * max(0.0, 1.0 - dist)
                    tipo = rng.choices([t for t, _ in MEZCLA], [p for _, p in MEZCLA])[0]

                    col_fac = rng.choice(colores_fachada)
                    col_tch = rng.choice(colores_techo)

                    hx0, hz0, hx1, hz1 = huella
                    esquinas = [(hx0, hz0), (hx1, hz0), (hx1, hz1), (hx0, hz1)]

                    pw, ph = min(2.5, (hx1 - hx0) * 0.35), 2.8
                    bld.add_caja(lcx - pw/2, 0.22, hz1 - 0.1, lcx + pw/2, 0.22 + ph, hz1 + 0.15,
                                 (50, 40, 30), (220, 180, 80), rotar=True)

                    if tipo == "caja":
                        bld.add_caja(hx0, 0, hz0, hx1, altura, hz1, col_fac, col_tch, rotar=True)
                        bld.add_caja(hx0, altura, hz0, hx1, altura + 1.2, hz1, col_fac, (30, 32, 36), rotar=True, con_borde=False)
                        cw, cd = (hx1 - hx0) * 0.35, (hz1 - hz0) * 0.35
                        bld.add_caja(lcx - cw/2, altura, lcz - cd/2, lcx + cw/2, altura + 3.5, lcz + cd/2,
                                     (70, 75, 82), (48, 50, 56), rotar=True)
                        self._agregar_ventanas(bld, hx0, hz0, hx1, hz1, altura, rng)

                    elif tipo == "escalonado":
                        y_cur = 0.0
                        niveles = rng.choice([2, 3])
                        fracs = [0.55, 0.45] if niveles == 2 else [0.45, 0.32, 0.23]
                        w_cur, d_cur = (hx1 - hx0), (hz1 - hz0)
                        for fr in fracs:
                            h_nv = altura * fr
                            x_0, x_1 = lcx - w_cur/2, lcx + w_cur/2
                            z_0, z_1 = lcz - d_cur/2, lcz + d_cur/2
                            bld.add_caja(x_0, y_cur, z_0, x_1, y_cur + h_nv, z_1, col_fac, col_tch, rotar=True)
                            self._agregar_ventanas(bld, x_0, z_0, x_1, z_1, h_nv, rng, y_offset=y_cur)
                            y_cur += h_nv
                            w_cur *= 0.72
                            d_cur *= 0.72
                        bld.add_piramide(lcx - w_cur/2, lcz - d_cur/2, lcx + w_cur/2, lcz + d_cur/2,
                                         y_cur, 8.0, (185, 155, 85), rotar=True)
                        bld.add_linea((lcx, y_cur + 8.0, lcz), (lcx, y_cur + 16.0, lcz), (220, 200, 120), rotar=True)

                    elif tipo == "gotica":
                        h_cuerpo = altura * 0.7
                        h_techo = max(10.0, min(32.0, altura * 0.35))
                        bld.add_caja(hx0, 0, hz0, hx1, h_cuerpo, hz1, col_fac, col_tch, rotar=True)
                        self._agregar_ventanas(bld, hx0, hz0, hx1, hz1, h_cuerpo, rng)
                        bld.add_piramide(hx0, hz0, hx1, hz1, h_cuerpo, h_techo, (45, 50, 60), rotar=True)
                        t_sz = min(3.0, (hx1 - hx0) * 0.22, (hz1 - hz0) * 0.22)
                        for tx, tz in ((hx0, hz0), (hx1 - t_sz, hz0), (hx0, hz1 - t_sz), (hx1 - t_sz, hz1 - t_sz)):
                            bld.add_caja(tx, h_cuerpo, tz, tx + t_sz, h_cuerpo + 3.0, tz + t_sz,
                                         col_fac, (45, 50, 60), rotar=True)
                            bld.add_piramide(tx, tz, tx + t_sz, tz + t_sz, h_cuerpo + 3.0, 7.0, (40, 45, 55), rotar=True)

                    else:
                        h_alm = rng.uniform(10, 16)
                        bld.add_caja(hx0, 0, hz0, hx1, h_alm, hz1, (105, 52, 42), (58, 60, 66), rotar=True)
                        bld.add_gable(hx0, hz0, hx1, hz1, h_alm, 5.0, (68, 72, 80), rotar=True)
                        chx, chz = hx1 - 2.5, hz0 + 2.0
                        bld.add_caja(chx - 0.9, 0, chz - 0.9, chx + 0.9, h_alm + 12.0, chz + 0.9,
                                     (120, 42, 32), (38, 38, 38), rotar=True)
                        for y_aro in (h_alm + 3, h_alm + 7, h_alm + 11):
                            bld.add_linea((chx - 1.0, y_aro, chz - 1.0), (chx + 1.0, y_aro, chz - 1.0), (30, 30, 30), rotar=True)
                            bld.add_linea((chx + 1.0, y_aro, chz - 1.0), (chx + 1.0, y_aro, chz + 1.0), (30, 30, 30), rotar=True)

                    edificios.append(Edificio(tipo, altura, esquinas))

        return edificios

    def _agregar_ventanas(self, bld, x0, z0, x1, z1, altura, rng, y_offset=0.0):
        w, d = x1 - x0, z1 - z0
        piso_h = 3.2
        pisos = max(1, int(altura / piso_h) - 1)
        col_calida = (255, 215, 120)
        col_azul = (165, 215, 245)
        col_oscura = (42, 48, 58)

        for p in range(pisos):
            y_win = y_offset + 2.2 + p * piso_h
            if y_win + 1.6 > y_offset + altura:
                break
            n_cols_x = max(1, int(w / 3.0))
            for k in range(n_cols_x):
                wx = x0 + 1.5 + k * (w - 3.0) / max(1, n_cols_x - 1) if n_cols_x > 1 else (x0 + x1) / 2
                cw = rng.choices([col_calida, col_azul, col_oscura], [0.35, 0.20, 0.45])[0]
                bld.add_quad((wx - 0.55, y_win, z1 + 0.05), (wx + 0.55, y_win, z1 + 0.05),
                             (wx + 0.55, y_win + 1.3, z1 + 0.05), (wx - 0.55, y_win + 1.3, z1 + 0.05),
                             cw, (0, 0, 1), rotar=True)
                bld.add_quad((wx + 0.55, y_win, z0 - 0.05), (wx - 0.55, y_win, z0 - 0.05),
                             (wx - 0.55, y_win + 1.3, z0 - 0.05), (wx + 0.55, y_win + 1.3, z0 - 0.05),
                             cw, (0, 0, -1), rotar=True)

            n_cols_z = max(1, int(d / 3.0))
            for k in range(n_cols_z):
                wz = z0 + 1.5 + k * (d - 3.0) / max(1, n_cols_z - 1) if n_cols_z > 1 else (z0 + z1) / 2
                cw = rng.choices([col_calida, col_azul, col_oscura], [0.35, 0.20, 0.45])[0]
                bld.add_quad((x1 + 0.05, y_win, wz - 0.55), (x1 + 0.05, y_win, wz + 0.55),
                             (x1 + 0.05, y_win + 1.3, wz + 0.55), (x1 + 0.05, y_win + 1.3, wz - 0.55),
                             cw, (1, 0, 0), rotar=True)
                bld.add_quad((x0 - 0.05, y_win, wz + 0.55), (x0 - 0.05, y_win, wz - 0.55),
                             (x0 - 0.05, y_win + 1.3, wz - 0.55), (x0 - 0.05, y_win + 1.3, wz + 0.55),
                             cw, (-1, 0, 0), rotar=True)

    def _wayne_solido(self, cx, cz, bld):
        f = (cx - 11, cz - 9, cx + 11, cz + 9)
        esq = [(f[0], f[1]), (f[2], f[1]), (f[2], f[3]), (f[0], f[3])]
        niveles = [
            (11, 9, 35.0, (135, 130, 120)),
            (8.5, 7.0, 30.0, (150, 140, 125)),
            (6.0, 5.0, 22.0, (165, 150, 130)),
            (4.0, 3.5, 13.0, (185, 165, 125)),
        ]
        y_cur = 0.0
        for hw, hd, h_tier, col in niveles:
            bld.add_caja(cx - hw, y_cur, cz - hd, cx + hw, y_cur + h_tier, cz + hd, col, (48, 50, 58), rotar=True)
            bld.add_caja(cx - hw - 0.3, y_cur + h_tier - 1.2, cz - hd - 0.3,
                         cx + hw + 0.3, y_cur + h_tier, cz + hd + 0.3,
                         (220, 185, 75), (240, 200, 85), rotar=True)
            self._agregar_ventanas(bld, cx - hw, cz - hd, cx + hw, cz + hd, h_tier, self.rng, y_offset=y_cur)
            y_cur += h_tier

        bld.add_piramide(cx - 3.0, cz - 2.5, cx + 3.0, cz + 2.5, y_cur, 14.0, (230, 195, 80), rotar=True)
        bld.add_caja(cx - 0.3, y_cur + 14.0, cz - 0.3, cx + 0.3, y_cur + 30.0, cz + 0.3,
                     (255, 230, 120), (255, 255, 255), rotar=True)
        bld.add_caja(cx - 0.6, y_cur + 30.0, cz - 0.6, cx + 0.6, y_cur + 31.2, cz + 0.6,
                     (255, 40, 40), (255, 60, 60), rotar=True)

        return Edificio("wayne", 100, esq, "Wayne Tower")

    def _arkham_solido(self, cx, cz, bld):
        f = (cx - 19, cz - 10, cx + 19, cz + 10)
        esq = [(f[0], f[1]), (f[2], f[1]), (f[2], f[3]), (f[0], f[3])]
        col_piedra = (82, 85, 95)
        col_tejado = (45, 48, 58)

        bld.add_caja(f[0], 0, f[1], f[2], 14.0, f[3], col_piedra, col_tejado, rotar=True)
        bld.add_gable(f[0], f[1], f[2], f[3], 14.0, 7.0, col_tejado, rotar=True)
        self._agregar_ventanas(bld, f[0], f[1], f[2], f[3], 14.0, self.rng)

        for tx, tz in ((f[0], f[1]), (f[2] - 6.5, f[1]), (f[0], f[3] - 6.5), (f[2] - 6.5, f[3] - 6.5)):
            bld.add_caja(tx, 0, tz, tx + 6.5, 34.0, tz + 6.5, (72, 75, 85), col_tejado, rotar=True)
            bld.add_piramide(tx, tz, tx + 6.5, tz + 6.5, 34.0, 16.0, (38, 42, 50), rotar=True)
            bld.add_linea(((tx + tx + 6.5)/2, 50.0, (tz + tz + 6.5)/2),
                          ((tx + tx + 6.5)/2, 54.0, (tz + tz + 6.5)/2), (180, 185, 195), rotar=True)

        tc = (cx - 5.0, cz - 5.0, cx + 5.0, cz + 5.0)
        bld.add_caja(tc[0], 0, tc[1], tc[2], 44.0, tc[3], (68, 70, 80), col_tejado, rotar=True)
        bld.add_piramide(tc[0], tc[1], tc[2], tc[3], 44.0, 22.0, (32, 35, 44), rotar=True)
        bld.add_caja(cx - 1.5, 37.0, tc[1] - 0.2, cx + 1.5, 40.0, tc[1] + 0.1, (235, 220, 140), rotar=True)

        return Edificio("arkham", 66, esq, "Arkham Asylum")

    def _gcpd_solido(self, cx, cz, bld):
        f = (cx - 14, cz - 9, cx + 14, cz + 9)
        esq = [(f[0], f[1]), (f[2], f[1]), (f[2], f[3]), (f[0], f[3])]
        col_gcpd = (72, 85, 110)
        bld.add_caja(f[0], 0, f[1], f[2], 24.0, f[3], col_gcpd, (48, 52, 62), rotar=True)
        self._agregar_ventanas(bld, f[0], f[1], f[2], f[3], 24.0, self.rng)

        # Helipuerto
        bld.add_caja(cx - 6, 24.0, cz - 6, cx + 6, 24.2, cz + 6, (38, 42, 50), (45, 50, 58), rotar=True)
        bld.add_caja(cx - 2.5, 24.25, cz - 2.0, cx - 1.7, 24.25, cz + 2.0, (240, 210, 40), rotar=True)
        bld.add_caja(cx + 1.7, 24.25, cz - 2.0, cx + 2.5, 24.25, cz + 2.0, (240, 210, 40), rotar=True)
        bld.add_caja(cx - 2.0, 24.25, cz - 0.4, cx + 2.0, 24.25, cz + 0.4, (240, 210, 40), rotar=True)

        # Proyector de la Bat-Señal
        bld.add_caja(cx - 2.0, 24.0, cz + 4.0, cx + 2.0, 28.5, cz + 8.0, (55, 60, 70), (80, 85, 95), rotar=True)
        base = (cx, 28.5, cz + 6.0)
        yc, zc, R = 112.0, cz + 24.0, 16.0
        n = 16
        aro = [(cx + R * math.cos(2 * math.pi * k / n), yc + R * 0.8 * math.sin(2 * math.pi * k / n), zc)
               for k in range(n)]

        for k in range(n):
            k_next = (k + 1) % n
            bld.add_tri(base, aro[k], aro[k_next], (255, 245, 175, 0.16), transp=True, rotar=True)
            bld.add_linea(aro[k], aro[k_next], (255, 235, 140), rotar=True)
            if k % 4 == 0:
                bld.add_linea(base, aro[k], (255, 240, 160), rotar=True)

        # Nube iluminada
        for k in range(n):
            k_next = (k + 1) % n
            bld.add_tri((cx, yc, zc), aro[k], aro[k_next], (255, 245, 185, 0.40), transp=True, rotar=True)

        # Silueta de Batman en el centro del haz
        bat = [(-1.0, 0.25), (-0.8, 0.6), (-0.55, 0.45), (-0.4, 0.15), (-0.2, 0.2),
               (-0.12, 0.5), (0.0, 0.35), (0.12, 0.5), (0.2, 0.2), (0.4, 0.15),
               (0.55, 0.45), (0.8, 0.6), (1.0, 0.25), (0.7, -0.05), (0.4, -0.2),
               (0.15, -0.1), (0.0, -0.45), (-0.15, -0.1), (-0.4, -0.2), (-0.7, -0.05)]
        pts = [(cx + bx * 9.5, yc + by * 9.5, zc + 0.1) for bx, by in bat]
        for k in range(1, len(pts) - 1):
            bld.add_tri(pts[0], pts[k], pts[k + 1], (14, 15, 20, 0.95), transp=True, rotar=True)
        for k in range(len(pts)):
            bld.add_linea(pts[k], pts[(k + 1) % len(pts)], (25, 25, 32), rotar=True)

        return Edificio("caja", 112, esq, "GCPD (Bat-Señal)")

    def _elipse_celda(self, c, r, puntos=28):
        x0, z0, x1, z1 = self._rect_celda(c, r)
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        rx, rz = SEP_AV * 0.34, SEP_CALLE * 0.31
        return [
            (cx + rx * math.cos(2 * math.pi * k / puntos),
             cz + rz * math.sin(2 * math.pi * k / puntos))
            for k in range(puntos)
        ]

    def _recortar(self, ruta, radio_a, radio_b):
        ruta = list(ruta)
        def mover(p, q, radio):
            dx, dz = q[0] - p[0], q[1] - p[1]
            L = math.hypot(dx, dz) or 1.0
            dx, dz = dx / L, dz / L
            t = min(radio / max(abs(dx), abs(dz) or 1.0), 0.45 * L)
            return (p[0] + dx * t, p[1] + dz * t)
        inicio = mover(ruta[0], ruta[1], radio_a)
        fin = mover(ruta[-1], ruta[-2], radio_b)
        ruta[0], ruta[-1] = inicio, fin
        return ruta

    @staticmethod
    def _huella_libre(cx, cz, w, d, ejes):
        for _ in range(8):
            f = (cx - w / 2, cz - d / 2, cx + w / 2, cz + d / 2)
            libre = True
            for k in range(25):
                px = f[0] + (f[2] - f[0]) * (k % 5) / 4
                pz = f[1] + (f[3] - f[1]) * (k // 5) / 4
                for pa, pb, hw in ejes:
                    if Ciudad._dist_segmento((px, pz), pa, pb) < hw + 2.0:
                        libre = False
                        break
                if not libre:
                    break
            if libre and w >= 5 and d >= 5:
                return f
            w *= 0.85
            d *= 0.85
        return None

    @staticmethod
    def _dist_segmento(p, a, b):
        ax, az = a
        bx, bz = b
        dx, dz = bx - ax, bz - az
        L2 = dx * dx + dz * dz or 1.0
        t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - az) * dz) / L2))
        return math.hypot(p[0] - (ax + t * dx), p[1] - (az + t * dz))

    # Control de semáforos
    @staticmethod
    def estado_en(eje, t):
        ciclo = 2 * (T_VERDE + T_AMARILLO)
        t = t % ciclo
        if eje == "NS":
            t = (t + T_VERDE + T_AMARILLO) % ciclo
        if t < T_VERDE:
            return "verde"
        if t < T_VERDE + T_AMARILLO:
            return "amarillo"
        return "rojo"

    def actualizar(self, dt):
        self.tiempo += dt
        for sem in self.semaforos:
            sem.estado = self.estado_en(sem.eje, self.tiempo)

    def estado_semaforo(self, col, fila, eje):
        return self.estado_en(eje, self.tiempo)

    def pos_nodo(self, col, fila):
        return self.nodos[self.idx[(col, fila)]]

    def centro_celda(self, col, fila):
        esq = [self.pos_nodo(col + i, fila + j) for i in (0, 1) for j in (0, 1)]
        return (sum(p[0] for p in esq) / 4, sum(p[2] for p in esq) / 4)

    def centro_zona(self, nombre):
        pts = self.zonas[nombre]
        return (sum(p[0] for p in pts) / len(pts), sum(p[2] for p in pts) / len(pts))

    def resumen(self):
        return {
            "intersecciones": len(self.nodos),
            "tramos": len(self.aristas),
            "edificios": len(self.edificios),
            "semaforos": len(self.semaforos),
            "farolas": self.n_farolas,
            "arboles": self.n_arboles,
        }

    # ------------------------------------------------------------------
    # Renderizado OpenGL
    # ------------------------------------------------------------------
    def compilar_display_list(self):
        self._display_list = gl.glGenLists(1)
        gl.glNewList(self._display_list, gl.GL_COMPILE)

        # 1. Geometría sólida con iluminación
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)

        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [150.0, 280.0, 100.0, 0.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.88, 0.90, 0.96, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.38, 0.40, 0.46, 1.0])

        gl.glBegin(gl.GL_TRIANGLES)
        for i in range(0, len(self.arr_solidos), 10):
            x, y, z, nx, ny, nz, r, g, b, a = self.arr_solidos[i:i+10]
            gl.glNormal3f(nx, ny, nz)
            gl.glColor4f(r, g, b, a)
            gl.glVertex3f(x, y, z)
        gl.glEnd()

        # 2. Líneas arquitectónicas
        gl.glDisable(gl.GL_LIGHTING)
        gl.glLineWidth(1.1)
        gl.glBegin(gl.GL_LINES)
        for i in range(0, len(self.arr_lineas), 6):
            x, y, z, r, g, b = self.arr_lineas[i:i+6]
            gl.glColor3f(r, g, b)
            gl.glVertex3f(x, y, z)
        gl.glEnd()

        # 3. Transparencias (Bat-Señal)
        if self.arr_transp is not None and len(self.arr_transp) > 0:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
            gl.glDepthMask(gl.GL_FALSE)
            gl.glBegin(gl.GL_TRIANGLES)
            for i in range(0, len(self.arr_transp), 10):
                x, y, z, nx, ny, nz, r, g, b, a = self.arr_transp[i:i+10]
                gl.glColor4f(r, g, b, a)
                gl.glVertex3f(x, y, z)
            gl.glEnd()
            gl.glDepthMask(gl.GL_TRUE)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glEndList()

    def dibujar(self):
        """Dibuja la ciudad según el modo activo (hibrido, solido o alambre)."""
        if self.auto_reloj:
            ahora = time.monotonic()
            if self._t_ultimo is not None:
                self.actualizar(ahora - self._t_ultimo)
            self._t_ultimo = ahora

        if self.modo == "alambre":
            # Modo alambre tradicional
            gl.glDisable(gl.GL_LIGHTING)
            gl.glLineWidth(1.0)
            gl.glBegin(gl.GL_LINES)
            for i in range(0, len(self.arr_lineas), 6):
                x, y, z, r, g, b = self.arr_lineas[i:i+6]
                gl.glColor3f(r, g, b)
                gl.glVertex3f(x, y, z)
            gl.glEnd()
        else:
            # Modo sólido o híbrido
            if self._display_list is None:
                self.compilar_display_list()
            gl.glCallList(self._display_list)

        # Semáforos dinámicos (cambian con el reloj)
        gl.glDisable(gl.GL_LIGHTING)
        for sem, luces in self.luces_semaforos_dinamicas:
            for pos, norm, nombre_luz in luces:
                activo = (sem.estado == nombre_luz)
                if activo:
                    r, g, b = COLOR_SEMAFORO[nombre_luz]
                    gl.glColor3f(r / 255.0, g / 255.0, b / 255.0)
                    r_luz = RADIO_LUZ * 1.35
                else:
                    r, g, b = COLOR_APAGADO[nombre_luz]
                    gl.glColor3f(r / 255.0, g / 255.0, b / 255.0)
                    r_luz = RADIO_LUZ * 0.85

                gl.glBegin(gl.GL_TRIANGLE_FAN if activo else gl.GL_LINE_LOOP)
                gl.glVertex3f(pos[0], pos[1], pos[2])
                for j in range(9):
                    t = 2 * math.pi * j / 8
                    gl.glVertex3f(pos[0] + r_luz * math.cos(t) * -norm[2],
                                  pos[1] + r_luz * math.sin(t),
                                  pos[2] + r_luz * math.cos(t) * norm[0])
                gl.glEnd()
