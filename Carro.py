"""lógica de carros de la simulación
"""

import bisect
import math

PASOS_CURVA = 24      
RADIO_GIRO = 7.0     


def _unitario(dx, dz):
    L = math.hypot(dx, dz) or 1.0
    return dx / L, dz / L


def _dif_angulo(a, b):
    """Diferencia a -> b en el rango (-pi, pi]."""
    d = (b - a) % (2 * math.pi)
    return d - 2 * math.pi if d > math.pi else d


class _LineaCarril:
    """Línea recta del carril de UNA calle, en el sentido de viaje."""

    def __init__(self, inicio, fin):
        self.inicio = inicio
        self.fin = fin
        self.d = _unitario(fin[0] - inicio[0], fin[1] - inicio[1])
        self.rumbo = math.atan2(self.d[0], self.d[1])   # 0 = hacia +z


class Ruta:
    """Trayectoria de un carro: puntos, rumbo en cada punto y distancia acumulada."""

    def __init__(self, ciudad, nodos, carril=0, ciclica=False):
        self.ciclica = ciclica
        ids = [ciudad.idx[n] for n in nodos]
        if ciclica:
            tramos = [(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
        else:
            tramos = list(zip(ids[:-1], ids[1:]))

        lineas = [self._linea_carril(ciudad, a, b, carril) for a, b in tramos]
        n = len(lineas)

        esquinas = []
        n_union = n if ciclica else n - 1
        for i in range(n_union):
            esquinas.append(self._esquina(lineas[i], lineas[(i + 1) % n]))

        self.puntos = []
        self.rumbos = []
        self.tipos = []       

        def agregar(p, rumbo, tipo):
            if self.puntos and math.dist(p, self.puntos[-1]) < 1e-6:
                return
            if self.puntos:
                self.tipos.append(tipo)
            self.puntos.append(p)
            self.rumbos.append(rumbo)

        if ciclica and esquinas[-1] is not None:
            inicio = esquinas[-1]["pb"]
        else:
            inicio = lineas[0].inicio
        agregar(inicio, lineas[0].rumbo, "RECTO")

        for i, linea in enumerate(lineas):
            esq = esquinas[i] if i < len(esquinas) else None
            fin = esq["pa"] if esq else linea.fin
            agregar(fin, linea.rumbo, "RECTO")
            if esq:
                pa, pc, pb = esq["pa"], esq["pc"], esq["pb"]
                for k in range(1, PASOS_CURVA + 1):
                    t = k / PASOS_CURVA
                    x = (1 - t) ** 2 * pa[0] + 2 * (1 - t) * t * pc[0] + t ** 2 * pb[0]
                    z = (1 - t) ** 2 * pa[1] + 2 * (1 - t) * t * pc[1] + t ** 2 * pb[1]
                    tx = 2 * (1 - t) * (pc[0] - pa[0]) + 2 * t * (pb[0] - pc[0])
                    tz = 2 * (1 - t) * (pc[1] - pa[1]) + 2 * t * (pb[1] - pc[1])
                    agregar((x, z), math.atan2(tx, tz), "GIRANDO")

        self.acum = [0.0]
        for p, q in zip(self.puntos[:-1], self.puntos[1:]):
            self.acum.append(self.acum[-1] + math.dist(p, q))
        self.longitud = self.acum[-1]

    @staticmethod
    def _linea_carril(ciudad, a, b, carril):
        """Línea del carril (derecho) al ir del nodo a al nodo b."""
        for ar in ciudad.aristas:
            if (ar.a, ar.b) == (a, b):
                adelante = True
                break
            if (ar.a, ar.b) == (b, a):
                adelante = False
                break
        else:
            raise ValueError(f"No existe calle entre los nodos {a} y {b}")

        (ax, _, az), (bx, _, bz) = ar.ruta[0], ar.ruta[-1]
        ux, uz = _unitario(bx - ax, bz - az)
        nx, nz = -uz, ux                       # normal del sentido a -> b
        lista = ar.carriles_ab if adelante else ar.carriles_ba
        off = lista[min(carril, len(lista) - 1)]

        if adelante:
            inicio, fin = (ax, az), (bx, bz)
        else:
            inicio, fin = (bx, bz), (ax, az)
        inicio = (inicio[0] + nx * off, inicio[1] + nz * off)
        fin = (fin[0] + nx * off, fin[1] + nz * off)
        return _LineaCarril(inicio, fin)

    @staticmethod
    def _esquina(l1, l2):
        """Curva (Bézier cuadrática) que une el carril l1 con el carril l2."""
        cruz = l1.d[0] * l2.d[1] - l1.d[1] * l2.d[0]
        if abs(cruz) < 1e-6:
            if l1.d[0] * l2.d[0] + l1.d[1] * l2.d[1] > 0:
                return None                    # sigue derecho
            raise ValueError("La ruta incluye una vuelta en U (no soportada)")
        wx = l2.inicio[0] - l1.inicio[0]
        wz = l2.inicio[1] - l1.inicio[1]
        t = (wx * l2.d[1] - wz * l2.d[0]) / cruz
        pc = (l1.inicio[0] + t * l1.d[0], l1.inicio[1] + t * l1.d[1])
        pa = (pc[0] - l1.d[0] * RADIO_GIRO, pc[1] - l1.d[1] * RADIO_GIRO)
        pb = (pc[0] + l2.d[0] * RADIO_GIRO, pc[1] + l2.d[1] * RADIO_GIRO)
        return {"pa": pa, "pc": pc, "pb": pb}

    def punto_en(self, s):
        """Posición (x, z), rumbo (grados) y estado del segmento a la distancia s."""
        if self.ciclica:
            s %= self.longitud
        else:
            s = max(0.0, min(s, self.longitud))
        j = bisect.bisect_right(self.acum, s) - 1
        j = max(0, min(j, len(self.puntos) - 2))
        largo = self.acum[j + 1] - self.acum[j]
        f = (s - self.acum[j]) / largo if largo > 0 else 0.0
        p, q = self.puntos[j], self.puntos[j + 1]
        x = p[0] + (q[0] - p[0]) * f
        z = p[1] + (q[1] - p[1]) * f
        r0, r1 = self.rumbos[j], self.rumbos[j + 1]
        rumbo = r0 + _dif_angulo(r0, r1) * f
        return x, z, math.degrees(rumbo), self.tipos[j]


class Carro:
    """Agente carro: avanza por su ruta a velocidad constante."""

    def __init__(self, ruta, velocidad=10.0, color=0, desfase=0.0):
        self.ruta = ruta
        self.velocidad = velocidad
        self.color = color
        self.s = desfase              # distancia recorrida sobre la ruta
        self.estado = "RECTO"
        self.x = self.z = self.angulo = 0.0
        self.llanta_ang = 0.0         # rotación acumulada de las llantas (rodado)
        self.volante_ang = 0.0        # ángulo de dirección de llantas delanteras
        self._colocar()

    def _colocar(self):
        self.x, self.z, self.angulo, self.estado = self.ruta.punto_en(self.s)

    def actualizar(self, dt):
        ang_anterior = self.angulo
        ds = self.velocidad * dt
        self.s += ds
        if not self.ruta.ciclica and self.s >= self.ruta.longitud:
            self.s -= self.ruta.longitud       # llegó al final: reaparece al inicio
        self._colocar()

        # Rodado acumulado de las 4 llantas (radio base 0.707 * escala 0.45)
        radio_llanta = 0.707 * 0.45
        self.llanta_ang = (self.llanta_ang + math.degrees(ds / radio_llanta)) % 360.0

        # Giro de llantas delanteras en curvas
        if dt > 0 and self.estado == "GIRANDO":
            d_ang = _dif_angulo(math.radians(ang_anterior), math.radians(self.angulo))
            objetivo = max(-35.0, min(35.0, math.degrees(d_ang) * 2.5))
            self.volante_ang += (objetivo - self.volante_ang) * min(1.0, dt * 10.0)
        else:
            self.volante_ang += (0.0 - self.volante_ang) * min(1.0, dt * 8.0)


def crear_carros(ciudad):
    definiciones = [
        ([(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)], False, 1, 13.0, 0, 0.0),    # autopista, hacia el este
        ([(4, 1), (3, 1), (2, 1), (1, 1), (0, 1)], False, 1, 12.0, 1, 20.0),   # autopista, hacia el oeste
        ([(4, 0), (3, 0), (2, 0), (1, 0), (0, 0)], False, 0, 10.0, 2, 0.0),    # calle, hacia el oeste
        ([(0, 2), (1, 2), (1, 3), (0, 3)], True, 0, 10.0, 3, 0.0),     # vuelta a la manzana, giros a la derecha
        ([(0, 2), (1, 2), (1, 3), (0, 3)], True, 0, 10.0, 4, 49.0),    # segundo carro en el mismo circuito
        ([(2, 2), (2, 3), (3, 3), (3, 2)], True, 0, 10.0, 5, 0.0),     # vuelta a la manzana, giros a la izquierda
        ([(2, 2), (2, 3), (3, 3), (3, 2)], True, 0, 10.0, 0, 48.0),    # segundo carro en el mismo circuito
    ]
    carros = []
    for nodos, ciclica, carril, vel, color, desfase in definiciones:
        ruta = Ruta(ciudad, nodos, carril=carril, ciclica=ciclica)
        carros.append(Carro(ruta, velocidad=vel, color=color, desfase=desfase))
    return carros