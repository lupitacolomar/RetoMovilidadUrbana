
import math
import random
import time

import numpy as np
from OpenGL import GL as gl


# ---------------------------------------------------------------------------
# Parámetros de la zona
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
N_ARBOLES = 16        # árboles secos en Robinson Park

# Atributos por tipo de vía
# --- Tamaño del carro y de los carriles -----------------------------------
# Dimensiones de carroceria.obj (unidades de modelo, con swapyz): largo x ancho
CARRO_LARGO_MODELO = 9.85
CARRO_ANCHO_MODELO = 3.92
ESCALA_CARRO = 0.45                      # <- Scale que se le pasa a Carro(dim, vel, Scale)
CARRO_LARGO = CARRO_LARGO_MODELO * ESCALA_CARRO     # ~4.43
CARRO_ANCHO = CARRO_ANCHO_MODELO * ESCALA_CARRO     # ~1.76
FACTOR_CARRIL = 1.7                      # ancho de carril / ancho del carro
ANCHO_CARRIL = 3.0                       # ~1.7 x 1.76 = 3.0
CAMELLON = 1.2                           # separador central de la autopista
assert ANCHO_CARRIL >= 1.5 * CARRO_ANCHO, "el carro no cabe cómodo en el carril"

TIPOS = {
    "calle":     {"vel": 40, "carriles": 2, "camellon": 0.0},
    "avenida":   {"vel": 60, "carriles": 4, "camellon": 0.0},
    "autopista": {"vel": 80, "carriles": 4, "camellon": CAMELLON},
}
for _t in TIPOS.values():
    _t["carril"] = ANCHO_CARRIL
    _t["ancho"] = _t["carriles"] * ANCHO_CARRIL + _t["camellon"]   # 6 / 12 / 13.2

COLORES = {
    # vías
    "calle":      (105, 118, 140),
    "avenida":    (150, 178, 220),
    "autopista":  (235, 175, 70),
    "nodo":       (205, 205, 220),
    "zona":       (70, 150, 110),
    # edificios
    "caja":       (125, 135, 160),
    "escalonado": (160, 172, 205),
    "gotica":     (178, 150, 215),
    "almacen":    (105, 115, 128),
    "wayne":      (245, 215, 110),
    "arkham":     (225, 110, 110),
    # infraestructura
    "poste":      (190, 190, 205),
    "farola":     (255, 205, 110),
    "arbol":      (150, 120, 90),
    "senal":      (255, 238, 150),     # Bat-Señal
    "gcpd":       (110, 170, 235),
}

# Colores de los semáforos
COLOR_SEMAFORO = {
    "verde":    (60, 230, 110),
    "amarillo": (250, 220, 60),
    "rojo":     (240, 60, 60),
}
# Luz apagada (se sigue viendo para que se note el semáforo de 3 luces)
COLOR_APAGADO = {
    "verde":    (25, 85, 50),
    "amarillo": (95, 85, 25),
    "rojo":     (100, 32, 32),
}
ORDEN_LUCES = ("rojo", "amarillo", "verde")      # de arriba hacia abajo

# (tipo, probabilidad) de los edificios "normales"
MEZCLA = [("caja", 0.20), ("escalonado", 0.30), ("gotica", 0.38), ("almacen", 0.12)]


class Arista:
    """Tramo de calle entre dos intersecciones."""

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
        # Centro de cada carril (distancia al eje de la vía) de adentro hacia afuera.
        # Tránsito por la derecha: yendo a->b se usa el lado de normal negativa
        # (n = (-dz, dx)), yendo b->a el de normal positiva.
        centros = [self.camellon / 2 + self.ancho_carril * (k + 0.5)
                   for k in range(self.carriles // 2)]
        self.carriles_ab = [-c for c in centros]
        self.carriles_ba = [c for c in centros]
        self.ruta = []          # puntos 3D del eje de la calle
        self.longitud = 0.0


class Semaforo:
    """Semáforo de 3 luces (rojo arriba, amarillo en medio, verde abajo) de un
    acceso a una intersección.

    eje:    "EW" (este-oeste) o "NS" (norte-sur): los semáforos del mismo eje
            comparten fase.
    nodo:   índice de la intersección que controla.
    x, z:   posición del poste (retícula, sin girar).
    ux, uz: dirección por la que llegan los carros hacia el semáforo
            (sentido nodo -> afuera); las luces miran hacia esa dirección.
    estado: "verde" | "amarillo" | "rojo" (solo una luz encendida).
    """

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


class Ciudad:
    def __init__(self):
        self.idx = {}             # (col, fila) -> índice de nodo
        self.nodos = []           # (x, y, z) ya girados
        self.aristas = []
        self.edificios = []
        self.zonas = {}           # nombre -> polígono 3D (agua / parque)

        self.semaforos = []       # objetos Semaforo
        self.n_farolas = 0
        self.n_arboles = 0
        self.tiempo = 0.0         # reloj de los semáforos (s)
        self.auto_reloj = True    # True: render avanza el reloj solo
        self._t_ultimo = None
        self.luces = None         # (S, 3, 8, 3): 3 luces x 8 puntos por semáforo

        # Geometría de dibujo para OpenGL: 2 vértices por segmento
        self.vertices = None      # (N, 3) float32
        self.colores_v = None     # (N, 3) float32, componentes en [0, 1]

        self.rng = random.Random(SEMILLA)
        self._construir()

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------
    @staticmethod
    def _posicion_base(c, r):
        return (c * SEP_AV, (FILAS - 1 - r) * SEP_CALLE)   # fila 0 = norte

    @staticmethod
    def _rect_celda(c, r):
        x0, x1 = c * SEP_AV, (c + 1) * SEP_AV
        z0 = (FILAS - 1 - (r + 1)) * SEP_CALLE
        z1 = (FILAS - 1 - r) * SEP_CALLE
        return x0, z0, x1, z1

    def _construir(self):
        pos2d = []
        aristas = []

        # --- Nodos (todas las intersecciones de la retícula)
        for r in range(FILAS):
            for c in range(COLUMNAS):
                self.idx[(c, r)] = len(pos2d)
                pos2d.append(self._posicion_base(c, r))

        # --- Aristas de la retícula
        for r in range(FILAS):
            for c in range(COLUMNAS):
                a = self.idx[(c, r)]
                if c + 1 < COLUMNAS:        # calle (este-oeste)
                    if r == FILA_PRINCIPAL:  # avenida principal, recta
                        aristas.append(Arista(a, self.idx[(c + 1, r)],
                                              "autopista", NOMBRE_PRINCIPAL))
                    else:
                        aristas.append(Arista(a, self.idx[(c + 1, r)], "calle"))
                if r + 1 < FILAS:           # avenida (norte-sur)
                    aristas.append(Arista(a, self.idx[(c, r + 1)], "avenida"))

        rutas2d = [[pos2d[ar.a], pos2d[ar.b]] for ar in aristas]

        # --- Radio de cada intersección (según la vía más ancha que llega)
        radio = [0.0] * len(pos2d)
        for ar in aristas:
            radio[ar.a] = max(radio[ar.a], ar.ancho / 2 + 0.7)
            radio[ar.b] = max(radio[ar.b], ar.ancho / 2 + 0.7)

        segs = []   # (p1, p2, clave_de_color), puntos 3D en retícula

        # --- Vías: dos bordes paralelos por calle + cuadro en cada intersección
        for ar, ruta in zip(aristas, rutas2d):
            Q = self._recortar(ruta, radio[ar.a], radio[ar.b])
            normales = self._normales(Q)
            for lado in (+1, -1):
                borde = [
                    (q[0] + lado * n[0] * ar.ancho / 2, 0.0,
                     q[1] + lado * n[1] * ar.ancho / 2)
                    for q, n in zip(Q, normales)
                ]
                for p1, p2 in zip(borde[:-1], borde[1:]):
                    segs.append((p1, p2, ar.tipo))

        for i, (x, z) in enumerate(pos2d):
            h = radio[i]
            esq = [(x - h, 0.0, z - h), (x + h, 0.0, z - h),
                   (x + h, 0.0, z + h), (x - h, 0.0, z + h)]
            for k in range(4):
                segs.append((esq[k], esq[(k + 1) % 4], "nodo"))

        # --- Manzanas verdes (agua / parque)
        zonas2d = {}
        for (c, r), nombre in CELDAS_VERDES.items():
            zonas2d[nombre] = self._elipse_celda(c, r)
        for poligono in zonas2d.values():
            n = len(poligono)
            for k in range(n):
                a = (poligono[k][0], 0.0, poligono[k][1])
                b = (poligono[(k + 1) % n][0], 0.0, poligono[(k + 1) % n][1])
                segs.append((a, b, "zona"))

        # --- Árboles secos en Robinson Park
        self._arboles(zonas2d, segs)

        # --- Farolas y semáforos
        self._farolas(aristas, rutas2d, segs)
        cabezas = self._semaforos(aristas, rutas2d, radio, segs)

        # --- Edificios
        ejes = [(pos2d[ar.a], pos2d[ar.b], ar.ancho / 2) for ar in aristas]
        edificios = self._generar_edificios(ejes, segs)

        # --- Giro de toda la zona: centrar y rotar alrededor de Y
        xs = [p[0] for p in pos2d]
        zs = [p[1] for p in pos2d]
        self._cx, self._cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2

        def a3d(lista2d):
            return self._girar([(x, 0.0, z) for x, z in lista2d]).tolist()

        self.nodos = a3d(pos2d)

        for ar, ruta in zip(aristas, rutas2d):
            ar.ruta = a3d(ruta)
            ar.longitud = sum(math.dist(p, q) for p, q in zip(ar.ruta[:-1], ar.ruta[1:]))
        self.aristas = aristas

        for nombre, poligono in zonas2d.items():
            self.zonas[nombre] = a3d(poligono)

        for e in edificios:
            e.huella = a3d(e.huella2d)
        self.edificios = edificios

        # Buffers de OpenGL: vértices y un color por vértice
        flat = []
        colores = []
        for p1, p2, clave in segs:
            flat.extend([p1, p2])
            colores.append(COLORES[clave])
        self.vertices = np.ascontiguousarray(self._girar(flat), dtype=np.float32)
        self.colores_v = np.ascontiguousarray(
            np.repeat(np.array(colores, dtype=np.float32) / 255.0, 2, axis=0))

        # Luces de semáforo (se dibujan aparte: cambian de color)
        self.luces = self._girar(cabezas).reshape(len(self.semaforos), 3, 8, 3)
        self.actualizar(0.0)

    def _girar(self, puntos):
        """Centra la zona y la gira ROTACION grados alrededor de Y.
        Recibe una lista de (x, y, z) y devuelve un arreglo (N, 3)."""
        p = np.array(puntos, dtype=np.float64).reshape(-1, 3)
        x, z = p[:, 0] - self._cx, p[:, 2] - self._cz
        ang = math.radians(ROTACION)
        c, s = math.cos(ang), math.sin(ang)
        return np.stack([c * x + s * z, p[:, 1], -s * x + c * z], axis=1)

    # ------------------------------------------------------------------
    # Infraestructura urbana
    # ------------------------------------------------------------------
    def _arboles(self, zonas2d, segs):
        """Árboles secos (tronco + ramas) dentro de Robinson Park."""
        poligono = zonas2d.get("Robinson Park")
        if not poligono:
            return
        rng = random.Random(SEMILLA + 7)      # no altera la semilla de los edificios
        cx = sum(p[0] for p in poligono) / len(poligono)
        cz = sum(p[1] for p in poligono) / len(poligono)
        rx = max(p[0] for p in poligono) - cx
        rz = max(p[1] for p in poligono) - cz

        for _ in range(N_ARBOLES):
            while True:
                u, v = rng.uniform(-1, 1), rng.uniform(-1, 1)
                if u * u + v * v <= 0.78:
                    break
            x, z = cx + u * rx, cz + v * rz
            h = rng.uniform(7, 12)
            segs.append(((x, 0.0, z), (x, h, z), "arbol"))
            ang0 = rng.uniform(0, 2 * math.pi)
            for k, fy in enumerate((0.55, 0.75, 1.0)):
                a = ang0 + k * 2.1
                largo = rng.uniform(2.5, 4.5)
                y0 = h * fy
                p0 = (x, y0, z)
                p1 = (x + largo * math.cos(a), y0 + largo * 0.7, z + largo * math.sin(a))
                segs.append((p0, p1, "arbol"))
                b = a + rng.choice((-0.6, 0.6))      # ramita
                segs.append((p1, (p1[0] + 1.6 * math.cos(b), p1[1] + 1.4,
                                  p1[2] + 1.6 * math.sin(b)), "arbol"))
            self.n_arboles += 1

    def _farolas(self, aristas, rutas2d, segs):
        """Dos farolas por tramo, una a cada lado de la vía (alternadas)."""
        for ar, ruta in zip(aristas, rutas2d):
            (ax, az), (bx, bz) = ruta[0], ruta[-1]
            L = math.hypot(bx - ax, bz - az)
            dx, dz = (bx - ax) / L, (bz - az) / L
            nx, nz = -dz, dx
            for t, lado in ((1 / 3, +1), (2 / 3, -1)):
                px, pz = ax + (bx - ax) * t, az + (bz - az) * t
                x = px + lado * nx * (ar.ancho / 2 + 0.9)
                z = pz + lado * nz * (ar.ancho / 2 + 0.9)
                xa, za = x - lado * nx * 2.4, z - lado * nz * 2.4     # brazo hacia la calle
                segs.append(((x, 0.0, z), (x, ALTO_FAROLA, z), "poste"))
                segs.append(((x, ALTO_FAROLA, z), (xa, ALTO_FAROLA, za), "poste"))
                for p1, p2 in self._caja((xa - 0.5, za - 0.5, xa + 0.5, za + 0.5),
                                         ALTO_FAROLA - 1.0, ALTO_FAROLA):
                    segs.append((p1, p2, "farola"))
                self.n_farolas += 1

    def _semaforos(self, aristas, rutas2d, radio, segs):
        """Un semáforo de 3 luces por acceso de cada intersección (a la derecha
        de quien llega). Poste y caja van con la geometría fija; las 3 luces
        (octágonos de 8 puntos) se devuelven aparte porque cambian de color."""
        luces = []
        y_luz = [ALTO_POSTE + ALTO_CABEZA * f for f in (0.8, 0.5, 0.2)]   # rojo, amarillo, verde
        for ar, ruta in zip(aristas, rutas2d):
            (ax_, az_), (bx_, bz_) = ruta[0], ruta[-1]
            L = math.hypot(bx_ - ax_, bz_ - az_)
            dx, dz = (bx_ - ax_) / L, (bz_ - az_) / L
            eje = "EW" if abs(dx) > abs(dz) else "NS"

            for nodo, (px, pz), (ux, uz) in ((ar.a, (ax_, az_), (dx, dz)),
                                             (ar.b, (bx_, bz_), (-dx, -dz))):
                # (ux, uz): del nodo hacia afuera por este acceso. Quien llega
                # avanza en (-ux, -uz); su derecha es (-uz, ux).
                rx_, rz_ = -uz, ux
                x = px + ux * (radio[nodo] + 1.2) + rx_ * (ar.ancho / 2 + 1.0)
                z = pz + uz * (radio[nodo] + 1.2) + rz_ * (ar.ancho / 2 + 1.0)
                segs.append(((x, 0.0, z), (x, ALTO_POSTE, z), "poste"))

                # caja: 0.6 a cada lado en la dirección (ux, uz), 0.8 a lo ancho
                lx, lz = -uz, ux
                y0, y1 = ALTO_POSTE, ALTO_POSTE + ALTO_CABEZA
                esq = [(x + ux * 0.6 * sa + lx * 0.8 * sb, z + uz * 0.6 * sa + lz * 0.8 * sb)
                       for sa, sb in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
                for k in range(4):
                    e1, e2 = esq[k], esq[(k + 1) % 4]
                    segs.append(((e1[0], y0, e1[1]), (e2[0], y0, e2[1]), "poste"))
                    segs.append(((e1[0], y1, e1[1]), (e2[0], y1, e2[1]), "poste"))
                    segs.append(((e1[0], y0, e1[1]), (e1[0], y1, e1[1]), "poste"))

                # 3 luces en la cara que mira a los carros que llegan
                fx, fz = x + ux * 0.64, z + uz * 0.64
                for yc in y_luz:
                    for j in range(8):
                        t = 2 * math.pi * j / 8
                        luces.append((fx + lx * RADIO_LUZ * math.cos(t), yc + RADIO_LUZ * math.sin(t),
                                      fz + lz * RADIO_LUZ * math.cos(t)))

                col = next(c for (c, f), i in self.idx.items() if i == nodo)
                fila = next(f for (c, f), i in self.idx.items() if i == nodo)
                self.semaforos.append(Semaforo(nodo, eje, col, fila, x, z, ux, uz))
        return luces

    # ------------------------------------------------------------------
    # Control de semáforos (la simulación puede leer/cambiar esto)
    # ------------------------------------------------------------------
    @staticmethod
    def estado_en(eje, t):
        """Estado de un eje ("EW" o "NS") en el instante t del ciclo."""
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
        """Avanza el reloj de los semáforos dt segundos y actualiza colores."""
        self.tiempo += dt
        for sem in self.semaforos:
            sem.estado = self.estado_en(sem.eje, self.tiempo)

    def estado_semaforo(self, col, fila, eje):
        """Estado del semáforo de la intersección (col, fila) para un eje."""
        return self.estado_en(eje, self.tiempo)

    # ------------------------------------------------------------------
    # Edificios
    # ------------------------------------------------------------------
    def _generar_edificios(self, ejes, segs):
        rng = self.rng
        edificios = []
        centro_x = (COLUMNAS - 1) * SEP_AV / 2
        centro_z = (FILAS - 1) * SEP_CALLE / 2

        for r in range(FILAS - 1):
            for c in range(COLUMNAS - 1):
                if (c, r) in CELDAS_VERDES:
                    continue

                x0, z0, x1, z1 = self._rect_celda(c, r)
                cx, cz = (x0 + x1) / 2, (z0 + z1) / 2

                if (c, r) == CELDA_ARKHAM:
                    edificios.append(self._arkham(cx, cz, segs))
                    continue
                if (c, r) == CELDA_WAYNE:
                    edificios.append(self._wayne(cx, cz, segs))
                    continue
                if (c, r) == CELDA_GCPD:
                    edificios.append(self._gcpd(cx, cz, segs))
                    continue

                ix0, ix1 = x0 + MARGEN_X, x1 - MARGEN_X
                iz0, iz1 = z0 + MARGEN_Z, z1 - MARGEN_Z
                lw, ld = (ix1 - ix0) / 2, (iz1 - iz0) / 2

                # 2 x 2 lotes por manzana
                lotes = []
                for i in range(2):
                    for j in range(2):
                        if rng.random() < 0.08:       # algún lote vacío
                            continue
                        lotes.append((
                            ix0 + lw * (i + 0.5) + rng.uniform(-1.2, 1.2),
                            iz0 + ld * (j + 0.5) + rng.uniform(-1.0, 1.0),
                            (lw - 3.5) * rng.uniform(0.75, 1.0),
                            (ld - 3.5) * rng.uniform(0.75, 1.0)))

                for lcx, lcz, w, d in lotes:
                    if True:

                        huella = self._huella_libre(lcx, lcz, w, d, ejes)
                        if huella is None:
                            continue

                        # Más altos hacia el centro de la zona
                        dist = math.hypot((lcx - centro_x) / (centro_x or 1),
                                          (lcz - centro_z) / (centro_z or 1))
                        altura = rng.uniform(24, 50) + 45 * max(0.0, 1.0 - dist)

                        tipo = rng.choices([t for t, _ in MEZCLA],
                                           [p for _, p in MEZCLA])[0]
                        edificios.append(self._levantar(tipo, huella, altura, segs))
        return edificios

    @staticmethod
    def _huella_libre(cx, cz, w, d, ejes):
        """Devuelve (x0,z0,x1,z1) o None. Se encoge hasta no tocar ninguna vía
        (importante junto a la diagonal del Schwartz Bypass)."""
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
        L2 = dx * dx + dz * dz
        t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - az) * dz) / L2))
        return math.hypot(p[0] - (ax + t * dx), p[1] - (az + t * dz))

    def _levantar(self, tipo, f, altura, segs):
        if tipo == "caja":
            s = self._caja(f, 0, altura)
        elif tipo == "escalonado":
            s = self._escalonado(f, altura, niveles=self.rng.choice([2, 3]))
        elif tipo == "gotica":
            s = self._gotica(f, altura)
        else:
            altura = self.rng.uniform(9, 14)
            s = self._almacen(f, altura)

        for p1, p2 in s:
            segs.append((p1, p2, tipo))
        return Edificio(tipo, altura, self._esquinas(f))

    def _wayne(self, cx, cz, segs):
        """Wayne Tower: torre escalonada muy alta con aguja."""
        f = (cx - 11, cz - 9, cx + 11, cz + 9)
        s = self._escalonado(f, 100, niveles=4, antena=18, techo_punta=12)
        for p1, p2 in s:
            segs.append((p1, p2, "wayne"))
        return Edificio("wayne", 100, self._esquinas(f), "Wayne Tower")

    def _arkham(self, cx, cz, segs):
        """Arkham Asylum: bloque bajo y ancho con torres de esquina y torre central."""
        f = (cx - 19, cz - 10, cx + 19, cz + 10)
        x0, z0, x1, z1 = f
        s = self._caja(f, 0, 14)
        s += self._gable(f, 14, 6)
        for tx, tz in ((x0, z0), (x1 - 6, z0), (x0, z1 - 6), (x1 - 6, z1 - 6)):
            t = (tx, tz, tx + 6, tz + 6)
            s += self._caja(t, 0, 32)
            s += self._piramide(t, 32, 14)
        t = (cx - 4.5, cz - 4.5, cx + 4.5, cz + 4.5)
        s += self._caja(t, 0, 42)
        s += self._piramide(t, 42, 20)
        for p1, p2 in s:
            segs.append((p1, p2, "arkham"))
        return Edificio("arkham", 62, self._esquinas(f), "Arkham Asylum")

    def _gcpd(self, cx, cz, segs):
        """GCPD: edificio bajo con reflector en el techo que proyecta la Bat-Señal."""
        f = (cx - 14, cz - 9, cx + 14, cz + 9)
        s = [(p1, p2, "gcpd") for p1, p2 in self._caja(f, 0, 22)]
        s += [(p1, p2, "gcpd") for p1, p2 in self._gable(f, 22, 4)]

        # reflector sobre el techo
        lamp = (cx - 2, cz - 2, cx + 2, cz + 2)
        s += [(p1, p2, "senal") for p1, p2 in self._caja(lamp, 26, 30)]

        # haz de luz (cono) hacia un aro en el cielo, con el murciélago dentro
        base = (cx, 30.0, cz)
        yc, zc, R = 112.0, cz + 18.0, 15.0
        n = 12
        aro = [(cx + R * math.cos(2 * math.pi * k / n), yc + R * 0.8 * math.sin(2 * math.pi * k / n), zc)
               for k in range(n)]
        for k in range(n):
            s.append((aro[k], aro[(k + 1) % n], "senal"))
            if k % 2 == 0:
                s.append((base, aro[k], "senal"))

        bat = [(-1.0, 0.25), (-0.8, 0.6), (-0.55, 0.45), (-0.4, 0.15), (-0.2, 0.2),
               (-0.12, 0.5), (0.0, 0.35), (0.12, 0.5), (0.2, 0.2), (0.4, 0.15),
               (0.55, 0.45), (0.8, 0.6), (1.0, 0.25), (0.7, -0.05), (0.4, -0.2),
               (0.15, -0.1), (0.0, -0.45), (-0.15, -0.1), (-0.4, -0.2), (-0.7, -0.05)]
        pts = [(cx + bx * 9, yc + by * 9, zc) for bx, by in bat]
        for k in range(len(pts)):
            s.append((pts[k], pts[(k + 1) % len(pts)], "senal"))

        segs.extend(s)
        return Edificio("caja", 112, self._esquinas(f), "GCPD (Bat-Señal)")

    @staticmethod
    def _esquinas(f):
        x0, z0, x1, z1 = f
        return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]

    # ------------------------------------------------------------------
    # Primitivas de alambre: devuelven listas de segmentos 3D
    # ------------------------------------------------------------------
    @staticmethod
    def _caja(f, ya, yb):
        x0, z0, x1, z1 = f
        b = [(x0, ya, z0), (x1, ya, z0), (x1, ya, z1), (x0, ya, z1)]
        t = [(x0, yb, z0), (x1, yb, z0), (x1, yb, z1), (x0, yb, z1)]
        s = []
        for i in range(4):
            j = (i + 1) % 4
            s += [(b[i], b[j]), (t[i], t[j]), (b[i], t[i])]
        return s

    @staticmethod
    def _piramide(f, y, alto):
        x0, z0, x1, z1 = f
        base = [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
        punta = ((x0 + x1) / 2, y + alto, (z0 + z1) / 2)
        s = []
        for i in range(4):
            s.append((base[i], base[(i + 1) % 4]))
            s.append((base[i], punta))
        return s

    @staticmethod
    def _gable(f, y, alto):
        """Techo a dos aguas; la cumbrera corre por el lado largo."""
        x0, z0, x1, z1 = f
        zc, xc = (z0 + z1) / 2, (x0 + x1) / 2
        if (x1 - x0) >= (z1 - z0):
            a, b = (x0, y + alto, zc), (x1, y + alto, zc)
            return [(a, b),
                    ((x0, y, z0), a), ((x0, y, z1), a),
                    ((x1, y, z0), b), ((x1, y, z1), b)]
        a, b = (xc, y + alto, z0), (xc, y + alto, z1)
        return [(a, b),
                ((x0, y, z0), a), ((x1, y, z0), a),
                ((x0, y, z1), b), ((x1, y, z1), b)]

    def _escalonado(self, f, altura, niveles=3, antena=None, techo_punta=0):
        """Torre art déco: cada nivel es más angosto que el de abajo."""
        fracciones = {2: [0.65, 0.35], 3: [0.5, 0.3, 0.2],
                      4: [0.4, 0.28, 0.2, 0.12]}[niveles]
        x0, z0, x1, z1 = f
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        w, d = x1 - x0, z1 - z0

        s, y = [], 0.0
        for i, frac in enumerate(fracciones):
            k = 0.72 ** i
            nivel = (cx - w * k / 2, cz - d * k / 2, cx + w * k / 2, cz + d * k / 2)
            s += self._caja(nivel, y, y + altura * frac)
            y += altura * frac

        if techo_punta:
            s += self._piramide(nivel, y, techo_punta)
            y += techo_punta
        s.append(((cx, y, cz), (cx, y + (antena or max(6.0, altura * 0.18)), cz)))
        return s

    def _gotica(self, f, altura):
        """Torre gótica: cuerpo, techo muy inclinado y torrecillas en las esquinas."""
        x0, z0, x1, z1 = f
        hb = altura * 0.7
        techo = max(10.0, min(36.0, altura * 0.4))
        s = self._caja(f, 0, hb)
        s += self._piramide(f, hb, techo)

        t = min(3.5, (x1 - x0) * 0.22, (z1 - z0) * 0.22)
        for tx, tz in ((x0, z0), (x1 - t, z0), (x0, z1 - t), (x1 - t, z1 - t)):
            s += self._piramide((tx, tz, tx + t, tz + t), hb, 9)
        return s

    def _almacen(self, f, altura):
        """Bodega industrial con chimenea (Gotham es una ciudad de fábricas)."""
        x0, z0, x1, z1 = f
        chimenea = (x1 - 3.0, z0 + 1.0, x1 - 1.4, z0 + 2.6)
        return (self._caja(f, 0, altura) + self._gable(f, altura, 5)
                + self._caja(chimenea, altura, altura + 10))

    # ------------------------------------------------------------------
    # Utilidades de geometría (2D en retícula)
    # ------------------------------------------------------------------
    @staticmethod
    def _recortar(ruta, radio_a, radio_b):
        """Acorta los extremos de la calle para que no entre al cuadro
        de la intersección."""
        ruta = list(ruta)

        def mover(p, q, radio):
            dx, dz = q[0] - p[0], q[1] - p[1]
            L = math.hypot(dx, dz)
            dx, dz = dx / L, dz / L
            t = min(radio / max(abs(dx), abs(dz)), 0.45 * L)
            return (p[0] + dx * t, p[1] + dz * t)

        inicio = mover(ruta[0], ruta[1], radio_a)
        fin = mover(ruta[-1], ruta[-2], radio_b)
        ruta[0], ruta[-1] = inicio, fin
        return ruta

    @staticmethod
    def _normales(Q):
        def normal(p, q):
            dx, dz = q[0] - p[0], q[1] - p[1]
            L = math.hypot(dx, dz)
            return (-dz / L, dx / L)

        segs = [normal(p, q) for p, q in zip(Q[:-1], Q[1:])]
        out = [segs[0]]
        for a, b in zip(segs[:-1], segs[1:]):
            nx, nz = a[0] + b[0], a[1] + b[1]
            L = math.hypot(nx, nz) or 1.0
            out.append((nx / L, nz / L))
        out.append(segs[-1])
        return out

    def _elipse_celda(self, c, r, puntos=28):
        x0, z0, x1, z1 = self._rect_celda(c, r)
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        rx, rz = SEP_AV * 0.34, SEP_CALLE * 0.31
        return [
            (cx + rx * math.cos(2 * math.pi * k / puntos),
             cz + rz * math.sin(2 * math.pi * k / puntos))
            for k in range(puntos)
        ]

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------
    def pos_nodo(self, col, fila):
        return self.nodos[self.idx[(col, fila)]]

    def centro_celda(self, col, fila):
        """Centro (x, z) de una manzana, ya girado."""
        esq = [self.pos_nodo(col + i, fila + j) for i in (0, 1) for j in (0, 1)]
        return (sum(p[0] for p in esq) / 4, sum(p[2] for p in esq) / 4)

    def centro_zona(self, nombre):
        pts = self.zonas[nombre]
        return (
            sum(p[0] for p in pts) / len(pts),
            sum(p[2] for p in pts) / len(pts),
        )

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
    # Render (OpenGL)
    # ------------------------------------------------------------------
    def dibujar(self):
        """Dibuja la ciudad. Requiere un contexto OpenGL activo y la cámara
        (proyección y modelview) ya configurada por quien llama."""
        # Reloj de semáforos (si la simulación no lo maneja por su cuenta)
        if self.auto_reloj:
            ahora = time.monotonic()
            if self._t_ultimo is not None:
                self.actualizar(ahora - self._t_ultimo)
            self._t_ultimo = ahora

        # Geometría fija: todos los segmentos de una sola vez
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.vertices)
        gl.glColorPointer(3, gl.GL_FLOAT, 0, self.colores_v)
        gl.glDrawArrays(gl.GL_LINES, 0, len(self.vertices))
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

        # Luces de semáforo: 3 octágonos por semáforo; la luz activa va
        # rellena y brillante, las otras se ven apagadas (solo contorno)
        for i, sem in enumerate(self.semaforos):
            for k, nombre in enumerate(ORDEN_LUCES):
                P = self.luces[i, k]
                if sem.estado == nombre:
                    r, g, b = COLOR_SEMAFORO[nombre]
                    gl.glColor3f(r / 255.0, g / 255.0, b / 255.0)
                    gl.glBegin(gl.GL_TRIANGLE_FAN)
                    gl.glVertex3f(*P.mean(axis=0))
                    for p in P:
                        gl.glVertex3f(*p)
                    gl.glVertex3f(*P[0])
                    gl.glEnd()
                else:
                    r, g, b = COLOR_APAGADO[nombre]
                    gl.glColor3f(r / 255.0, g / 255.0, b / 255.0)
                    gl.glBegin(gl.GL_LINE_LOOP)
                    for p in P:
                        gl.glVertex3f(*p)
                    gl.glEnd()
