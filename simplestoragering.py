# Wei, Bingfeng
# wbf2016@mail.ustc.edu.cn
"""2020.12
"""

import copy
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.core.umath import sqrt, cos, sin, cosh, sinh, tan, pi
from scipy import constants, integrate, signal, fftpack
import matplotlib.pyplot as plt

LENGTH_PRECISION = 10  # 长度，消除浮点数的误差
Cr = 8.846E-5  # unit m/GeV3
Cq = 3.832E-13


class UnfinishedWork(Exception):
    """user's exception for marking the unfinished work"""

    def __init__(self, *args):
        self.name = args[0]

    # def __str__(self):
    #     print(str(self.name))


class ParticleLost(Exception):
    def __init__(self, *args):
        self.value = args


class Particle(object):
    """set particle's type and energy"""
    energy = None
    gamma = None
    beta = None

    @classmethod
    def set_energy(cls, energy, particle_type: str = "electron"):
        """energy: MeV"""
        cls.energy = energy
        if particle_type != 'electron':
            raise UnfinishedWork()
        text = particle_type + " mass energy equivalent in MeV"
        __emass = constants.physical_constants[text][0]
        cls.gamma = cls.energy / __emass
        cls.beta = sqrt(1 - 1 / cls.gamma ** 2)

    @classmethod
    def __str__(cls):
        return "mass: %s , gamma = %s" % (cls.energy / cls.gamma, cls.gamma)


class Step(object):
    """set calculation step for each component

    set calculation for each type or subtype,

    the rule of symbol:
    hundred place: type of components
        1: drift, 2: dipole, 3: quadrupole, 4: sextupole,
    ten place: subtype of components
        21: sector dipole, 22: rectangle dipole, 24: dipole edge
        32: normal focus, 31: normal defocus
        05: rf cavity
    one place: specific component
        241: dipole inlet edge, 242: dipole outlet edge"""

    def __init__(self, step):
        assert isinstance(step, dict)
        self.step = step

    def match(self, symbol: int):
        try:
            return self.step[symbol]
        except KeyError:
            try:
                return self.step[10 * int(symbol / 10)]
            except KeyError:
                try:
                    return self.step[100 * int(symbol / 100)]
                except KeyError:
                    try:
                        return self.step[0]
                    except KeyError:
                        raise Exception("can't find step of symbol %s" % symbol)

    def __str__(self):
        return str(self.step)


# components and line of components


class Components(metaclass=ABCMeta):
    """Parent class of magnet components"""
    symbol = None
    name = None
    length = 0
    h = 0
    k1 = 0
    k2 = 0
    aperture = np.inf

    def sub_matrix(self, direction):
        if direction == 'x':
            return np.array([[self.matrix[0, 0], self.matrix[0, 1]],
                             [self.matrix[1, 0], self.matrix[1, 1]]])
        elif direction == 'y':
            return np.array([[self.matrix[2, 2], self.matrix[2, 3]],
                             [self.matrix[3, 2], self.matrix[3, 3]]])
        else:
            raise Exception("direction must be 'x' or 'y' !!!")

    @property
    @abstractmethod
    def matrix(self):
        pass

    @property
    def matrix_eta(self):
        return np.array([self.matrix[0, 5], self.matrix[1, 5]])

    @abstractmethod
    def slice_matrix(self, new_length):
        pass

    def set_aperture(self, aperture):
        self.aperture = aperture

    def __str__(self):
        return "%s, %s, %s, %s, %s\n" % (self.name, self.length, self.h, self.k1, self.k2)

    def __repr__(self):
        return self.name


class UdfTransMatrix(Components):
    """generate a transfer matrix for non-specific component"""

    def __init__(self, matrix, length):
        assert matrix.shape == (6, 6), "matrix must be 6*6 numpy.ndarray"
        self._matrix = matrix
        self.length = length

    @property
    def matrix(self):
        return self._matrix

    def slice_matrix(self, new_length):
        raise Exception("can't renew a non-specific component")

    def __str__(self):
        return "length = %s \n matrix = \n%s" % (self.length, self.matrix)


class Drift(Components):
    symbol = 100

    def __init__(self, name: str, length: float):
        self.name = name
        self.length = length
        self.para_max = None
        self.para_min = None

    @property
    def matrix(self):
        return np.array([[1., self.length, 0, 0, 0, 0],
                         [0, 1., 0, 0, 0, 0],
                         [0, 0, 1., self.length, 0, 0],
                         [0, 0, 0, 1., 0, 0],
                         [0, 0, 0, 0, 1., self.length / Particle.gamma ** 2],
                         [0, 0, 0, 0, 0, 1]])

    def slice_matrix(self, new_length):
        return Drift(self.name, new_length)

    def set_limit(self, min_length, max_length):
        self.para_max = max_length
        self.para_min = min_length

    def rela_adjust(self, length_rela_change):
        new_length = self.length * length_rela_change
        if self.para_max is not None:
            assert new_length <= self.para_max, "adjusted length of %s is too long!" % self.name
        if self.para_min is not None:
            assert new_length >= self.para_min, "adjusted length of %s is too short!" % self.name
        new_drift = Drift(self.name, new_length)
        new_drift.set_limit(self.para_min, self.para_max)
        return new_drift

    def __str__(self):
        return "Drift:%s length = %s \n" % \
               (self.name, self.length)

    def __repr__(self):
        return "\nDrift:    %s\n    length = %s" % (self.name, self.length)


class DipoleEdge(Components):
    """generate a dipole edge component"""
    symbol = 240
    length = 0

    def __init__(self, name, rho, theta_e):
        self.name = name
        self.rho = rho
        self.theta_e = theta_e
        self.h = 1 / self.rho

    @property
    def matrix(self):
        return np.array([[1, 0, 0, 0, 0, 0],
                         [tan(self.theta_e) / self.rho, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, -tan(self.theta_e) / self.rho, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])

    def slice_matrix(self, new_length):
        raise Exception


class Dipole(Components):
    """generate a dipole component

    Attribute:
        name
        length
        theta: radian
        inlet: inlet edge
        outlet: outlet edge
        matrix: transfer matrix
    """
    symbol = 200

    def __init__(self, name: str, length: float, theta: float, theta_in: float, theta_out: float, k=0.0):
        self.name = name
        self.length = length
        self.theta = theta
        self.rho = self.length / self.theta
        self.h = 1 / self.rho
        self.para_min = None
        self.para_max = None
        self.k1 = k
        self.inlet = DipoleEdge(name + '_inlet', self.rho, theta_in)
        self.outlet = DipoleEdge(name + '_outlet', self.rho, theta_out)
        if theta_in == theta_out == 0:
            self.symbol = 210
        elif theta_in == theta / 2 and theta_out == theta / 2:
            self.symbol = 220

    @property
    def matrix(self):
        # TODO: 二四极组合
        if self.k1 == 0:
            sector_matrix = np.array([[cos(self.theta), self.rho * sin(self.theta), 0, 0, 0,
                                       self.rho * (1 - cos(self.theta))],
                                      [-sin(self.theta) / self.rho, cos(self.theta), 0, 0, 0, sin(self.theta)],
                                      [0, 0, 1, self.rho * self.theta, 0, 0],
                                      [0, 0, 0, 1, 0, 0],
                                      [-sin(self.theta), -self.rho * (1 - cos(self.theta)), 0, 0, 1,
                                       self.rho * (sin(self.theta) - Particle.beta ** 2 * self.theta)],
                                      [0, 0, 0, 0, 0, 1]])
        else:
            fx = self.k1 + self.h ** 2
            [cx, sx, dx] = self.__calculate_csd(fx)
            if fx != 0:
                m56 = self.length / Particle.gamma ** 2 - self.h ** 2 * (self.length - sx) / fx
            else:
                m56 = self.length / Particle.gamma ** 2 - self.h ** 2 * self.length ** 3 / 6
            fy = - self.k1
            [cy, sy, dy] = self.__calculate_csd(fy)
            sector_matrix = np.array([[cx, sx, 0, 0, 0, self.h * dx],
                                      [-fx * sx, cx, 0, 0, 0, self.h * sx],
                                      [0, 0, cy, sy, 0, 0],
                                      [0, 0, -fy * sy, cy, 0, 0],
                                      [- self.h * sx, - self.h * dx, 0, 0, 1, m56],
                                      [0, 0, 0, 0, 0, 1]])
        return self.outlet.matrix.dot(sector_matrix).dot(self.inlet.matrix)

    def __calculate_csd(self, fu):
        if fu > 0:
            sqrt_fu_z = sqrt(fu) * self.length
            cu = cos(sqrt_fu_z)
            su = sin(sqrt_fu_z) / sqrt(fu)
            du = (1 - cu) / fu
        elif fu < 0:
            sqrt_fu_z = sqrt(-fu) * self.length
            cu = cosh(sqrt_fu_z)
            su = sinh(sqrt_fu_z)
            du = (1 - cu) / fu
        else:
            cu = 1
            su = self.length
            du = self.length ** 2 / 2
        return [cu, su, du]

    def slice_matrix(self, new_length):
        new_theta = new_length / self.rho
        return Dipole(self.name, new_length, new_theta, 0, 0, self.k1)

    def rela_adjust(self, l_rela_change):
        new_length = self.length * l_rela_change
        if self.para_max is not None:
            assert new_length <= self.para_max, "the adjusted length of %s is too long!" % self.name
        if self.para_min is not None:
            assert new_length >= self.para_min, "the adjusted length of %s is too short!" % self.name
        new_dipole = Dipole(self.name, new_length, self.theta, self.inlet.theta_e, self.outlet.theta_e, self.k1)
        new_dipole.set_limit(self.para_min, self.para_max)
        return new_dipole

    def set_limit(self, length_min, length_max):
        self.para_min = length_min
        self.para_max = length_max

    def slice_inlet(self, new_length):
        """return a component with inlet edge"""
        new_theta = new_length / self.rho
        return Dipole(self.name, new_length, new_theta, self.inlet.theta_e, 0, self.k1)

    def slice_outlet(self, new_length):
        new_theta = new_length / self.rho
        return Dipole(self.name, new_length, new_theta, 0, self.outlet.theta_e, self.k1)

    def __str__(self):
        return "Dipole:%s rho = %s, theta = %s , theta_in = %s , theta_out = %s \n" % \
               (self.name, self.rho, self.theta, self.inlet.theta_e, self.outlet.theta_e)

    def __repr__(self):
        return "\nDipole:    %s\n    length = %s, theta(D) = %s" % \
               (self.name, self.length, round(self.theta * 180 / pi, 2))


class Quadrupole(Components):
    symbol = 300

    def __init__(self, name: str, length, k1):
        self.name = name
        self.length = length
        self.k1 = k1
        self.para_max = None
        self.para_min = None
        if k1 > 0:
            self.symbol = 320
        else:
            self.symbol = 310

    @property
    def matrix(self):
        if self.k1 > 0:
            sqk = sqrt(self.k1)
            sqkl = sqk * self.length
            return np.array([[cos(sqkl), sin(sqkl) / sqk, 0, 0, 0, 0],
                             [- sqk * sin(sqkl), cos(sqkl), 0, 0, 0, 0],
                             [0, 0, cosh(sqkl), sinh(sqkl) / sqk, 0, 0],
                             [0, 0, sqk * sinh(sqkl), cosh(sqkl), 0, 0],
                             [0, 0, 0, 0, 1, self.length / Particle.gamma ** 2],
                             [0, 0, 0, 0, 0, 1]])
        else:
            sqk = sqrt(-self.k1)
            sqkl = sqk * self.length
            return np.array([[cosh(sqkl), sinh(sqkl) / sqk, 0, 0, 0, 0],
                             [sqk * sinh(sqkl), cosh(sqkl), 0, 0, 0, 0],
                             [0, 0, cos(sqkl), sin(sqkl) / sqk, 0, 0],
                             [0, 0, - sqk * sin(sqkl), cos(sqkl), 0, 0],
                             [0, 0, 0, 0, 1, self.length / Particle.gamma ** 2],
                             [0, 0, 0, 0, 0, 1]])

    def slice_matrix(self, new_length):
        return Quadrupole(self.name, new_length, self.k1)

    def rela_adjust(self, k1_rela_change):
        new_k1 = self.k1 * k1_rela_change
        if self.para_max is not None:
            assert new_k1 <= self.para_max, "the adjusted k1 of %s exceeds the upper limit!" % self.name
        if self.para_min is not None:
            assert new_k1 >= self.para_min, "adjusted k1 of %s is too small!" % self.name
        new_qua = Quadrupole(self.name, self.length, new_k1)
        new_qua.set_limit(self.para_min, self.para_max)
        return new_qua

    def set_limit(self, min_k1, max_k1):
        self.para_min = min_k1
        self.para_max = max_k1

    def __str__(self):
        return "Quadrupole:%s length = %s, K = %s \n" % \
               (self.name, self.length, self.k1)

    def __repr__(self):
        return "\nQuadrupole:    %s\n    length = %s, k1 = %s" % (self.name, self.length, self.k1)


class Sextupole(Components):
    """generate a sextupole component, the 1st order transfer map is drift"""
    symbol = 400

    def __init__(self, name: str, length, k2):
        self.name = name
        self.length = length
        self.k2 = k2
        self.para_max = None
        self.para_min = None
        if k2 > 0:
            self.symbol = 420
        else:
            self.symbol = 410

    @property
    def matrix(self):
        return np.array([[1., self.length, 0, 0, 0, 0],
                         [0, 1., 0, 0, 0, 0],
                         [0, 0, 1., self.length, 0, 0],
                         [0, 0, 0, 1., 0, 0],
                         [0, 0, 0, 0, 1., self.length / Particle.gamma ** 2],
                         [0, 0, 0, 0, 0, 1]])

    def slice_matrix(self, new_length):
        return Sextupole(self.name, new_length, self.k2)

    def set_limit(self, min_k2, max_k2):
        self.para_min = min_k2
        self.para_max = max_k2

    def rela_adjust(self, k2_rela_change):
        new_k2 = self.k2 * k2_rela_change
        if self.para_max is not None:
            assert new_k2 <= self.para_max, "the adjusted k2 of %s exceeds the upper limit!" % self.name
        if self.para_min is not None:
            assert new_k2 >= self.para_min, "adjusted k2 of %s is too small!" % self.name
        new_sext = Sextupole(self.name, self.length, new_k2)
        new_sext.set_limit(self.para_min, self.para_max)
        return new_sext

    def __repr__(self):
        return "\nSextupole:    %s\n    length = %s, k2 = %s" % (self.name, self.length, self.k2)


class CombQuadSext(Components):
    symbol = 500

    def __init__(self, name: str, length, k1, k2):
        self.name = name
        self.length = length
        self.k1 = k1
        self.k2 = k2
        self.para_max = None
        self.para_min = None
        if k1 > 0:
            self.symbol = 520
        else:
            self.symbol = 510

    @property
    def matrix(self):
        if self.k1 > 0:
            sqk = sqrt(self.k1)
            sqkl = sqk * self.length
            return np.array([[cos(sqkl), sin(sqkl) / sqk, 0, 0, 0, 0],
                             [- sqk * sin(sqkl), cos(sqkl), 0, 0, 0, 0],
                             [0, 0, cosh(sqkl), sinh(sqkl) / sqk, 0, 0],
                             [0, 0, sqk * sinh(sqkl), cosh(sqkl), 0, 0],
                             [0, 0, 0, 0, 1, self.length / Particle.gamma ** 2],
                             [0, 0, 0, 0, 0, 1]])
        else:
            sqk = sqrt(-self.k1)
            sqkl = sqk * self.length
            return np.array([[cosh(sqkl), sinh(sqkl) / sqk, 0, 0, 0, 0],
                             [sqk * sinh(sqkl), cosh(sqkl), 0, 0, 0, 0],
                             [0, 0, cos(sqkl), sin(sqkl) / sqk, 0, 0],
                             [0, 0, - sqk * sin(sqkl), cos(sqkl), 0, 0],
                             [0, 0, 0, 0, 1, self.length / Particle.gamma ** 2],
                             [0, 0, 0, 0, 0, 1]])

    def slice_matrix(self, new_length):
        return CombQuadSext(self.name, new_length, self.k1, self.k2)

    def set_limit(self, min_k2, max_k2):
        self.para_min = min_k2
        self.para_max = max_k2

    def rela_adjust(self, k2_rela_change):
        new_k2 = self.k2 * k2_rela_change
        if self.para_max is not None:
            assert new_k2 <= self.para_max, "the adjusted k2 of %s exceeds the upper limit!" % self.name
        if self.para_min is not None:
            assert new_k2 >= self.para_min, "adjusted k2 of %s is too small!" % self.name
        new_sext = Sextupole(self.name, self.length, new_k2)
        new_sext.set_limit(self.para_min, self.para_max)
        return new_sext

    def __repr__(self):
        return "\nSextupole:    %s\n    length = %s, k1 = %s, k2 = %s" % (self.name, self.length, self.k1, self.k2)


class RFCavity(Components):
    symbol = 999

    def __init__(self):
        raise UnfinishedWork


class Line(Components):
    """generate a line of components

        Attributes:
            components: list of components
            length: length of line
        """

    def __init__(self, iterable):
        self.components = []
        self.length = 0
        for g in iterable:
            self.components.append(g)
        for g in self.components:
            self.length = round(self.length + g.length, LENGTH_PRECISION)

    @property
    def matrix(self):
        matrix = np.identity(6)
        for g in self.components:
            matrix = g.matrix.dot(matrix)
        return matrix

    @property
    def is_linear(self):
        for com in self.components:
            if com.symbol >= 400:
                return False
        return True

    def set_aperture(self, aperture, checking=1):
        for comp in self.components:
            comp.aperture = min(comp.aperture, aperture)
            if checking:
                print(str(comp.name) + ' aperture is set to ' + str(comp.aperture))

    def find_component(self, zi) -> (Components, float):
        """find the component at zi, return the component and the remaining length

        find the component after zi, if zi is at the end of the line, return the last component
        """
        g = None
        for g in self.components:
            zi = round(zi - g.length, LENGTH_PRECISION)
            if zi < 0:
                break
        return g, -zi

    def generate_matrix(self, z1: float, z0: float = 0) -> UdfTransMatrix:
        """generate transfer matrix ( <UdfTransferMatrix> ) from z0 (default 0) to z1"""
        assert 0 <= z0 <= z1 <= self.length, "0 <= z0 < z1 <= line length"
        new_length = z1 - z0
        comp_iter = iter(self.components)
        while True:
            component = next(comp_iter)
            z0 = round(z0 - component.length, LENGTH_PRECISION)
            if z0 < 0:
                break
        z1 = round(new_length + z0, LENGTH_PRECISION)
        if z1 <= 0:
            new_matrix = component.slice_matrix(new_length).matrix
            return UdfTransMatrix(new_matrix, new_length)
        else:
            if isinstance(component, Dipole):
                new_matrix = component.slice_outlet(-z0).matrix
            else:
                new_matrix = component.slice_matrix(-z0).matrix
            while True:
                component = next(comp_iter)
                z1 = round(z1 - component.length, LENGTH_PRECISION)
                if z1 <= 0:
                    break
                new_matrix = component.matrix.dot(new_matrix)
            if isinstance(component, Dipole):
                new_matrix = component.slice_inlet(component.length + z1).matrix.dot(new_matrix)
            else:
                new_matrix = component.slice_matrix(component.length + z1).matrix.dot(new_matrix)
            return UdfTransMatrix(new_matrix, new_length)

    def step_matrix(self, z0, step, z_last):
        """generate sliced components step by step, yield (sliced_comp, zi at the end of the matrix, zi is edge? )"""
        assert isinstance(step, Step)
        while z0 < z_last:
            distance = z_last - z0
            magnet, remain_length = self.find_component(z0)
            is_edge = 0
            if isinstance(magnet, Dipole):  # 二极铁，要考虑边缘场
                distance = min(distance, step.match(magnet.symbol))  # 距离为设定步长与终点距离的较小值
                if remain_length == magnet.length:  # 如果起点在磁铁入口，要考虑入口边缘场
                    if distance >= magnet.length:  # 如果距离超过了磁铁长度，返回磁铁的传输矩阵
                        z0 = round(z0 + magnet.length, LENGTH_PRECISION)
                        is_edge = 1
                        yield magnet, z0, is_edge
                    else:  # 如果没有到磁铁出口，返回带入口边缘场的传输矩阵
                        z0 = round(z0 + distance, LENGTH_PRECISION)
                        yield magnet.slice_inlet(distance), z0, is_edge
                elif remain_length <= distance:  # 如果没有经过入口，而经过了出口，只考虑出口边缘场
                    z0 = round(z0 + remain_length, LENGTH_PRECISION)
                    is_edge = 1
                    yield magnet.slice_outlet(remain_length), z0, is_edge
                else:  # 既不经过出口，又不经过入口，不用考虑边缘场
                    z0 = round(z0 + distance, LENGTH_PRECISION)
                    yield magnet.slice_matrix(distance), z0, is_edge
            else:  # 不是二极铁，不考虑边缘场
                length = min(remain_length, distance)
                local_step = step.match(magnet.symbol)
                if length > local_step:
                    z0 = round(z0 + local_step, LENGTH_PRECISION)
                    yield magnet.slice_matrix(local_step), z0, is_edge
                else:
                    z0 = round(z0 + length, LENGTH_PRECISION)
                    is_edge = 1
                    yield magnet.slice_matrix(length), z0, is_edge

    def z_axis_generator(self, z0, step: Step, z_last):
        """generate zi, same as step_matrix but don't yield matrix"""
        assert isinstance(step, Step), "step must be a dict {'drift': , 'dipole': ,...}"
        while z0 < z_last:
            distance = z_last - z0
            magnet, remain_length = self.find_component(z0)
            length = min(remain_length, distance)
            local_step = step.match(magnet.symbol)
            if length > local_step:
                z0 = round(z0 + local_step, LENGTH_PRECISION)
                is_edge = 0
                yield z0, is_edge
            else:
                z0 = round(z0 + length, LENGTH_PRECISION)
                is_edge = 1
                yield z0, is_edge

    def adjust(self, component, precision):
        """adjust the parameter of component and return a new Line

        adjust length of Drift or length of Dipole or k1 of Quadrupole relatively (1e-6)"""
        adjust_relatively = 1 + precision
        temp_list = []
        for comp in self.components:
            if component is comp:
                new_comp = comp.rela_adjust(adjust_relatively)
            else:
                new_comp = copy.deepcopy(comp)
            temp_list.append(new_comp)
        return Line(temp_list)

    def show_matrix(self):
        matrix = np.eye(6)
        for comp in self.components:
            print(comp.name)
            print(comp.matrix)
            matrix = comp.matrix.dot(matrix)
            print('Concatenated matrix after last element:')
            print(matrix)
            print('\n')
        print('full matrix')
        print(self.matrix)

    def slice_matrix(self, new_length):
        raise Exception("can't slice a line")

    def __str__(self):
        return " line: %s \n length = %s \n matrix: \n%s" % (self.components, self.length, self.matrix)


class Element(object):
    """slice class, containing magnet information and beam or particle data, and containing methods to calculate

    because of function 'find_component', the data in a element is the data at the beginning of the element.
    so when use the data and the length of element to integral, it would be better to integrate forwards"""

    def __init__(self):
        # basic attributes, must define
        self.z_axis = None
        self.length = None
        self.symbol = None
        self.matrix = None
        self.k1 = 0
        self.k2 = 0
        self.h = 0
        self.aperture = np.inf
        # special attributes for edge
        self.theta_e = None
        # attributes for lattice
        self.beta_x = None
        self.alpha_x = None
        self.gamma_x = None
        self.dmux = None
        self.mux = None
        self.nux = None
        self.beta_y = None
        self.alpha_y = None
        self.gamma_y = None
        self.dmuy = None
        self.muy = None
        self.nuy = None
        self.eta = None
        self.etap = None
        self.curl_H = None
        # attributes for track
        self.particle = None

    def get_magnet_data(self, magnet):
        """h k1 k2 aperture"""
        self.h = magnet.h
        self.k1 = magnet.k1
        self.k2 = magnet.k2
        self.aperture = magnet.aperture

    def sub_matrix(self, direction):
        """return sub_matrix of x or y direction"""
        if direction == 'x':
            return np.array([[self.matrix[0, 0], self.matrix[0, 1]],
                             [self.matrix[1, 0], self.matrix[1, 1]]])
        elif direction == 'y':
            return np.array([[self.matrix[2, 2], self.matrix[2, 3]],
                             [self.matrix[3, 2], self.matrix[3, 3]]])
        else:
            raise Exception("direction must be 'x' or 'y' !!!")

    def __get_twiss(self, direction):
        if direction == 'x':
            return np.array([self.beta_x, self.alpha_x, self.gamma_x])
        elif direction == 'y':
            return np.array([self.beta_y, self.alpha_y, self.gamma_y])

    def next_twiss(self, direction):
        """calculate twiss parameters at the element's exit according to the data at the element's entrance"""
        sub = self.sub_matrix(direction)
        matrix_cal = np.array([[sub[0, 0] ** 2, -2 * sub[0, 0] * sub[0, 1], sub[0, 1] ** 2],
                               [-sub[0, 0] * sub[1, 0], 2 * sub[0, 1] * sub[1, 0] + 1, -sub[0, 1] * sub[1, 1]],
                               [sub[1, 0] ** 2, -2 * sub[1, 0] * sub[1, 1], sub[1, 1] ** 2]])
        return matrix_cal.dot(self.__get_twiss(direction))

    def next_eta_bag(self):
        """calculate  parameters at the element's exit according to the data at the element's entrance"""
        eta_bag = np.array([self.eta, self.etap])
        return self.sub_matrix('x').dot(eta_bag) + np.array([self.matrix[0, 5], self.matrix[1, 5]])

    def next_particle(self):
        """calculate the particle position at the exit according to the data at the entrance
        """
        if self.k2 == 0 and self.k1 == 0:  # warning: don't use splitting method on Dipole,
            # because edge fields has not been considered
            new_particle = self.__linear_next_particle()
        else:
            try:
                new_particle = self.___drift(self.___kick(self.___drift(self.particle, self.length / 2), self.length),
                                             self.length / 2)
            except ParticleLost:
                raise ParticleLost
        return new_particle

    def ___kick(self, particle, ds):
        [x1, px1, y1, py1, z1, delta1] = particle
        px2 = px1 - ds * (self.k1 * x1 + self.k2 * (x1 ** 2 - y1 ** 2) / 2)
        py2 = py1 + ds * (self.k1 * y1 + self.k2 * x1 * y1)
        delta2 = delta1
        return np.array([x1, px2, y1, py2, z1, delta2])

    def ___drift(self, particle, ds):
        [x0, px0, y0, py0, z0, delta0] = particle
        d_sqare = 1 - px0 ** 2 - py0 ** 2 + delta0 ** 2 + 2 * delta0 / Particle.beta
        if d_sqare < 0:
            raise ParticleLost
        d0 = sqrt(d_sqare)
        x1 = x0 + ds * px0 / d0
        y1 = y0 + ds * py0 / d0
        z1 = z0
        return np.array([x1, px0, y1, py0, z1, delta0])

    def __linear_next_particle(self):
        new_particle = self.matrix.dot(self.particle)
        return new_particle

    def __str__(self):
        val = ('symbol = ' + str(self.symbol) + '    z = ' + str(self.z_axis) + '    length = ' + str(self.length))
        if self.h != 0:
            val += ('    h = ' + str(self.h))
        if self.k1 != 0:
            val += ('    k1 = ' + str(self.k1))
        if self.k2 != 0:
            val += ('    k2 = ' + str(self.k2))
        return val


# storage ring


class VirtualRFCavity(object):
    """a virtual rf cavity, don't have length, just for energy calculation

    Attributes:
        voltage: total voltage in MeV
    """

    def __init__(self, voltage, harmonic_number):
        self.voltage = voltage  # set to MeV, because U0 is MeV
        self.harmonic_number = harmonic_number
        self.phase = None
        self.omega_rf = None
        self.synchrotron_tune = None


def lattice_twiss(matrix):
    cos_mu = (matrix[0, 0] + matrix[1, 1]) / 2
    assert abs(cos_mu) < 1, '运动不稳定，无周期解'
    mu = np.arccos(cos_mu) * np.sign(matrix[0, 1])
    beta = matrix[0, 1] / sin(mu)
    alpha = (matrix[0, 0] - matrix[1, 1]) / (2 * sin(mu))
    gamma = - matrix[1, 0] / sin(mu)
    return np.array([beta, alpha, gamma])


class Lattice(object):
    """generate a lattice, and calculate the parameters"""

    def __init__(self, line: Line, calculate_step: Step, periods_number: int, coupling=0.00):
        assert isinstance(line, Line)
        assert isinstance(calculate_step, Step)
        self.line = line
        self.step = calculate_step
        self.periods_number = periods_number
        self.coupl = coupling
        self.elements = self.ring_element()
        # parameters needs integration
        self.xi_x = None
        self.xi_y = None
        self.I1 = None
        self.I2 = None
        self.I3 = None
        self.I4 = None
        self.I5 = None
        self.radiation_integrals()
        # global parameters
        self.Jx = None
        self.Jy = None
        self.Js = None
        self.sigma_e = None
        self.emittance = None
        self.U0 = None
        self.Tperiod = None
        self.tau0 = None
        self.tau_s = None
        self.tau_x = None
        self.tau_y = None
        self.alpha = None
        self.emitt_x = None
        self.emitt_y = None
        self.etap = None
        self.sigma_z = None
        self.global_parameters()
        self.rf_cavity = None

    @property
    def length(self):
        return self.line.length * self.periods_number

    @property
    def nux(self):
        return self.elements[-1].nux * self.periods_number

    @property
    def nuy(self):
        return self.elements[-1].nuy * self.periods_number

    def set_virtual_rf_cavity(self, voltage, harmonic_number):
        """set virtual rf cavity, length and location will not be considered"""
        self.rf_cavity = VirtualRFCavity(voltage, harmonic_number)
        self.rf_cavity.omega_rf = harmonic_number * 2 * pi / self.Tperiod
        phase = np.arcsin(self.U0 / self.rf_cavity.voltage)  # this result is in [0, pi / 2]
        if self.etap < 0:
            self.rf_cavity.phase = phase
        else:
            self.rf_cavity.phase = pi - phase
        self.rf_cavity.synchrotron_tune = sqrt(self.rf_cavity.voltage * self.rf_cavity.omega_rf * abs(
            cos(self.rf_cavity.phase) * self.etap) * self.length / Particle.energy / constants.c) / 2 / pi
        print('successfully set rf cavity!')
        print('    rf cavity phase is ' + str(round(self.rf_cavity.phase * 180 / pi, 2)) + '°')
        print('    Synchrotron tune is ' + str(self.rf_cavity.synchrotron_tune))
        self.cal_sigma_z()

    def cal_sigma_z(self):
        if self.rf_cavity:
            # k_z_square = (self.rf_cavity.voltage * cos(self.rf_cavity.phase) * self.etap *
            #               self.rf_cavity.omega_rf / (Particle.energy * self.length * constants.c))
            # self.sigma_z = self.sigma_e * abs(self.etap) / sqrt(abs(k_z_square))
            self.sigma_z = self.sigma_e * abs(self.etap) * self.length / (2 * pi * self.rf_cavity.synchrotron_tune)
        else:
            raise UnfinishedWork('calculate sigma_z with a real rf cavity')

    def initial_eta_bag(self):
        eta_bag = np.linalg.inv(np.identity(2) - self.line.sub_matrix('x')).dot(self.line.matrix_eta)
        return eta_bag

    def initial_twiss(self, direction):
        return lattice_twiss(self.line.sub_matrix(direction))

    def ring_element(self):
        """generate list of elements along the ring

        Watch Out!
        Theoretically, the code doesn't consider the situation that the beginning or ending component is DIPOLE,
        but usually this won't happen, the beginning and ending should be drift"""
        elements = []
        ele = Element()
        ele.z_axis = 0
        magnet, drop_data = self.line.find_component(0)
        ele.symbol = magnet.symbol
        ele.theta_e = 0
        [beta_x, alpha_x, gamma_x] = self.initial_twiss('x')
        [beta_y, alpha_y, gamma_y] = self.initial_twiss('y')
        [eta, etap] = self.initial_eta_bag()
        mux = 0
        muy = 0
        ele.mux = mux
        ele.muy = muy
        ele.nux = ele.mux / (2 * pi)
        ele.nuy = ele.muy / (2 * pi)
        ele.beta_y = beta_y
        ele.alpha_y = alpha_y
        ele.gamma_y = gamma_y
        ele.beta_x = beta_x
        ele.alpha_x = alpha_x
        ele.gamma_x = gamma_x
        ele.eta = eta
        ele.etap = etap
        ele.curl_H = ele.gamma_x * ele.eta ** 2 + 2 * ele.alpha_x * ele.eta * ele.etap + ele.beta_x * ele.etap ** 2
        for slice_comp, zi, is_edge in self.line.step_matrix(0, self.step, self.line.length):
            ele.length = round(zi - ele.z_axis, LENGTH_PRECISION)  # between z_i and z_(i-1)
            ele.matrix = slice_comp.matrix
            ele.dmux = np.arctan(slice_comp.matrix[0, 1] / (slice_comp.matrix[0, 0] * beta_x -
                                                            slice_comp.matrix[0, 1] * alpha_x))
            ele.dmuy = np.arctan(slice_comp.matrix[2, 3] / (slice_comp.matrix[2, 2] * beta_y -
                                                            slice_comp.matrix[2, 3] * alpha_y))
            if is_edge and 200 <= ele.symbol < 300:
                ele.symbol = 242
                last_magnet, drop_data = self.line.find_component(round(ele.z_axis, LENGTH_PRECISION))
                ele.theta_e = last_magnet.outlet.theta_e
            elements.append(copy.deepcopy(ele))
            [beta_x, alpha_x, gamma_x] = ele.next_twiss('x')
            [beta_y, alpha_y, gamma_y] = ele.next_twiss('y')
            [eta, etap] = ele.next_eta_bag()
            # data of ele_i
            ele.z_axis = zi
            magnet, drop_data = self.line.find_component(ele.z_axis)
            ele.get_magnet_data(magnet)
            if is_edge and 200 <= magnet.symbol < 300:
                ele.symbol = 241
                ele.theta_e = magnet.inlet.theta_e
            else:
                ele.symbol = magnet.symbol
                ele.theta_e = 0
            ele.beta_y = beta_y
            ele.alpha_y = alpha_y
            ele.gamma_y = gamma_y
            ele.beta_x = beta_x
            ele.alpha_x = alpha_x
            ele.gamma_x = gamma_x
            ele.eta = eta
            ele.etap = etap
            ele.curl_H = ele.gamma_x * ele.eta ** 2 + 2 * ele.alpha_x * ele.eta * ele.etap + ele.beta_x * ele.etap ** 2
            mux = mux + ele.dmux
            muy = muy + ele.dmuy
            ele.mux = mux
            ele.muy = muy
            ele.nux = ele.mux / (2 * pi)
            ele.nuy = ele.muy / (2 * pi)
        ele.length = 0
        ele.dmux = 0
        ele.dmuy = 0
        ele.matrix = np.identity(6)
        elements.append(copy.deepcopy(ele))
        return elements

    def radiation_integrals(self):
        integral1 = 0
        integral2 = 0
        integral3 = 0
        integral4 = 0
        integral5 = 0
        chromaticity_x = 0
        chromaticity_y = 0
        for ele in self.elements:
            integral1 = integral1 + ele.length * ele.eta * ele.h
            integral2 = integral2 + ele.length * ele.h ** 2
            integral3 = integral3 + ele.length * abs(ele.h) ** 3
            integral4 = integral4 + ele.length * (ele.h ** 2 + 2 * ele.k1) * ele.eta * ele.h
            if ele.symbol == 241:
                integral4 = integral4 + ele.h ** 2 * ele.eta * tan(ele.theta_e)
            elif ele.symbol == 242:
                integral4 = integral4 - ele.h ** 2 * ele.eta * tan(ele.theta_e)
            integral5 = integral5 + ele.length * ele.curl_H * abs(ele.h) ** 3
            chromaticity_x = chromaticity_x - ((ele.k1 + ele.h ** 2 - ele.eta * ele.k2) * ele.length
                                               - ele.h * tan(ele.theta_e)) * ele.beta_x
            chromaticity_y = chromaticity_y + ((ele.k1 - ele.eta * ele.k2) * ele.length
                                               - ele.h * tan(ele.theta_e)) * ele.beta_y
        self.I1 = integral1 * self.periods_number
        self.I2 = integral2 * self.periods_number
        self.I3 = integral3 * self.periods_number
        self.I4 = integral4 * self.periods_number
        self.I5 = integral5 * self.periods_number
        self.xi_x = chromaticity_x * self.periods_number / (4 * pi)
        self.xi_y = chromaticity_y * self.periods_number / (4 * pi)
        return 0

    def global_parameters(self):
        self.Jx = 1 - self.I4 / self.I2
        self.Jy = 1
        self.Js = 2 + self.I4 / self.I2
        self.sigma_e = Particle.gamma * sqrt(Cq * self.I3 / (self.Js * self.I2))
        self.emittance = Cq * Particle.gamma * Particle.gamma * self.I5 / (self.Jx * self.I2)
        self.U0 = 1e3 * Cr * (Particle.energy * 1e-3) ** 4 * self.I2 / (2 * pi)
        self.Tperiod = self.line.length * self.periods_number / (constants.c * Particle.beta)
        self.tau0 = 2 * Particle.energy * self.Tperiod / self.U0
        self.tau_s = self.tau0 / self.Js
        self.tau_x = self.tau0 / self.Jx
        self.tau_y = self.tau0 / self.Jy
        self.alpha = self.I1 / (constants.c * self.Tperiod)  # momentum compaction factor
        self.emitt_x = self.emittance / (1 + self.coupl)
        self.emitt_y = self.emittance * self.coupl / (1 + self.coupl)
        self.etap = self.alpha - 1 / Particle.gamma ** 2  # phase slip factor
        return 0

    def lifetime(self):
        raise UnfinishedWork('call touschek_lifetime')

    def touschek_lifetime_opa(self, beam_current_mA, bunches):
        """Unfinished work"""

        def linear_delta_acc(curl_H):
            list_acc = []
            for temp_ele in self.elements:
                list_acc.append(temp_ele.aperture / (sqrt(curl_H * temp_ele.beta_x) + temp_ele.eta))
            return min(list_acc)


        def nonlinear_delta_acc():
            raise UnfinishedWork('nonlinear')

        def touschek_function():
            f1 = lambda u: np.exp(- zeta / u) / u
            f2 = lambda u: np.log(u) * np.exp(- zeta / u) / 2
            f3 = lambda u: - np.exp(- zeta / u)
            term1 = integrate.quad(f1, 0, 1)[0]
            term2 = integrate.quad(f2, 0, 1)[0]
            term3 = integrate.quad(f3, 0, 1)[0]
            Dfunction = term1 + term2 + term3
            return Dfunction

        re = 2.8179403227E-15  # m
        # raise UnfinishedWork
        plot_data = {}
        list_z = []
        list_sigma_x = []
        list_sigma_y = []
        list_volume = []
        list_delta_acc = []
        list_zeta = []
        list_loss_rate = []
        list_xp = []
        list_temp = []
        # file1 = open('touschek_plot.txt', 'w')
        # file1.write("s betax betay disp curlH sigmax sigmay acc zeta")
        charge_per_bunch_nC = beam_current_mA * self.Tperiod * 1e6 / bunches
        print('charge per bunch: %s  nC' % charge_per_bunch_nC)
        Neb = charge_per_bunch_nC / 1.602176634e-10
        if self.sigma_z is None:
            self.cal_sigma_z()
        int_rate = 0
        # if self.line.is_linear:
        #     delta_acc = linear_delta_acc(self.elements[0].curl_H)
        # else:
        #     delta_acc = nonlinear_delta_acc()
        delta_acc = linear_delta_acc(self.elements[0].curl_H)
        for ele in self.elements:
            if ele.h != 0:
                # if self.line.is_linear:
                #     delta_acc = linear_delta_acc(ele.curl_H)
                # else:
                #     delta_acc = nonlinear_delta_acc()
                delta_acc = linear_delta_acc(ele.curl_H)
            sigma_x = sqrt(ele.beta_x * self.emitt_x + ele.eta ** 2 * self.sigma_e ** 2)
            sigma_xp = (self.emitt_x / sigma_x) * sqrt(1 + ele.curl_H * self.sigma_e ** 2 / self.emitt_x)
            sigma_y = sqrt(ele.beta_y * self.emitt_y)
            zeta = (delta_acc / Particle.gamma / sigma_xp) ** 2
            # zeta = (delta_acc * ele.beta_x / Particle.gamma / sigma_x) ** 2
            d_value = touschek_function()
            loss_rate = re ** 2 * constants.c * Neb * d_value / (sigma_x * sigma_y * sigma_xp * delta_acc ** 2 * 8 * pi
                                                                 * Particle.gamma ** 3 * self.sigma_z)
            int_rate += loss_rate * ele.length
            list_z.append(ele.z_axis)
            list_sigma_x.append(sigma_x)
            list_sigma_y.append(sigma_y)
            list_delta_acc.append(delta_acc)
            list_zeta.append(zeta)
            list_volume.append(sigma_x * sigma_y * self.sigma_z)
            list_loss_rate.append(loss_rate)
            list_xp.append(sigma_xp)
            list_temp.append(sigma_x / ele.beta_x)
            # file1.write("\n%s %s %s %s %s %s %s %s %s %s" % (ele.z_axis, ele.beta_x, ele.beta_y, ele.eta, ele.curl_H,
            #                                               sigma_x, sigma_y, delta_acc, zeta, loss_rate))
        average_rate = int_rate / self.line.length
        touschek_second = 1 / average_rate
        # file1.close()
        print('touschek lifetime is ' + str(touschek_second / 3600) + ' h')
        plot_data['s'] = list_z
        plot_data['sigma_x'] = list_sigma_x
        plot_data['sigma_y'] = list_sigma_y
        plot_data['delta_acc'] = list_delta_acc
        plot_data['zeta'] = list_zeta
        plot_data['volume'] = list_volume
        plot_data['loss_rate'] = list_loss_rate
        plot_data['sigmaxp'] = list_xp
        return 1 / average_rate, plot_data

    def touschek_lifetime(self, beam_current_mA, bunches):
        def calculate_delta_acc():
            list_acc = []
            for temp_ele in self.elements:
                if temp_ele.etap != 0:
                    list_acc.append(0.02 / 2 / abs(temp_ele.etap))
            return np.min(list_acc)

        def func_zeta():
            def f2(x):
                return np.log(x) * np.exp(- x) / x

            term2 = integrate.quad(f2, zeta, np.inf)[0] * 0.5 * zeta
            f3 = lambda x: np.exp(-x) / x
            term3 = integrate.quad(f3, zeta, np.inf)[0] * 0.5 * (3 * zeta - zeta * np.log(zeta) + 2)
            term1 = -1.5 * np.exp(-zeta)
            Dfunction = np.sqrt(zeta) * (term1 + term2 + term3)
            return Dfunction

        re = 2.8179403227E-15  # m
        charge_per_bunch_nC = beam_current_mA * self.Tperiod * 1e6 / bunches
        print('charge per bunch: %s  nC' % charge_per_bunch_nC)
        Neb = charge_per_bunch_nC / 1.602176634e-10
        if self.sigma_z is None:
            self.cal_sigma_z()
        delta_acc = calculate_delta_acc()
        inte_rate = 0
        plot_data = {}
        list_z = []
        list_loss_rate = []
        list_xp = []
        list_zeta = []
        list_difference_zeta = []
        list_acc = []
        for ele in self.elements:
            sigma_x = sqrt(ele.beta_x * self.emitt_x + ele.eta ** 2 * self.sigma_e ** 2)
            sigma_xp = (self.emitt_x / sigma_x) * sqrt(1 + ele.curl_H * self.sigma_e ** 2 / self.emitt_x)
            sigma_y = sqrt(ele.beta_y * self.emitt_y)
            zeta = (delta_acc * ele.beta_x / Particle.gamma / sigma_x) ** 2
            d_value = func_zeta()
            tau_touschek = (8 * constants.pi * sigma_x * sigma_y * self.sigma_z * Particle.gamma ** 2 * delta_acc ** 3
                            / d_value / Neb / re ** 2 / constants.c)
            inte_rate = inte_rate + ele.length / tau_touschek
            list_z.append(ele.z_axis)
            list_loss_rate.append(1 / tau_touschek)
            list_xp.append(sigma_xp)
            list_difference_zeta.append(sigma_xp * ele.beta_x / sigma_x)
            list_zeta.append(zeta)
            list_acc.append(delta_acc)
        average_tau = 1 / inte_rate * self.line.length
        print('touschek lifetime is ' + str(average_tau / 3600) + ' h')
        plot_data['s'] = list_z
        plot_data['loss_rate'] = list_loss_rate
        plot_data['sigmaxp'] = list_xp
        plot_data['difference_zeta'] = list_difference_zeta
        plot_data['zeta'] = list_zeta
        plot_data['acc'] = list_acc
        # plt.plot(list_z, list_loss_rate)
        # plt.ylabel('1/(s m)')
        # plt.xlabel('m')
        # plt.title('loss rate')
        # plt.show()
        return average_tau, plot_data

    def tune_adjust(self, tune_x, tune_y):
        """relatively smooth tuning of the lattice within a small range

        gathers quadrupoles in two groups depending on the sign of the k-value, and establishes a sensitivity matrix
        for a relative change of strength (dk/k = 1e-6).
        """

        def tune_sensitivity_matrix():
            rela_matrix = np.zeros([2, 2])
            test_k = 1e-6
            temp_comps = []
            for comp in self.line.components:
                if 320 <= comp.symbol < 330:
                    temp_comp = Quadrupole(comp.name, comp.length, comp.k1 * (1 + test_k))
                else:
                    temp_comp = copy.deepcopy(comp)
                temp_comps.append(temp_comp)
            temp_line = Line(temp_comps)
            temp_lattice = Lattice(temp_line, self.step, self.periods_number, self.coupl)
            rela_matrix[0, 0] = temp_lattice.nux - self.nux
            rela_matrix[1, 0] = temp_lattice.nuy - self.nuy
            temp_comps2 = []
            for comp in self.line.components:
                if 310 <= comp.symbol < 320:
                    temp_comp = Quadrupole(comp.name, comp.length, comp.k1 * (1 + test_k))
                else:
                    temp_comp = copy.deepcopy(comp)
                temp_comps2.append(temp_comp)
            temp_line = Line(temp_comps2)
            temp_lattice = Lattice(temp_line, self.step, self.periods_number, self.coupl)
            rela_matrix[0, 1] = temp_lattice.nux - self.nux
            rela_matrix[1, 1] = temp_lattice.nuy - self.nuy
            return np.linalg.inv(rela_matrix) * test_k

        delta_tunex = tune_x - self.elements[-1].nux * self.periods_number
        delta_tuney = tune_y - self.elements[-1].nuy * self.periods_number
        sensi_matrix = tune_sensitivity_matrix()
        [delta_kp, delta_km] = sensi_matrix.dot(np.array([delta_tunex, delta_tuney]))
        new_components = []
        for component in self.line.components:
            if 320 <= component.symbol < 330:
                new_comp = component.rela_adjust(1 + delta_kp)
            elif 310 <= component.symbol < 320:
                new_comp = component.rela_adjust(1 + delta_km)
            else:
                new_comp = copy.deepcopy(component)
            new_components.append(new_comp)
        new_line = Line(new_components)
        new_lattice = Lattice(new_line, self.step, self.periods_number, self.coupl)
        if abs(new_lattice.nux - tune_x) > 1e-3 or abs(new_lattice.nuy - tune_y) > 1e-3:
            return new_lattice.tune_adjust(tune_x, tune_y)
        else:
            print("\n[dk/k F, dk/k D] = sensitivity_matrix.dot([dnux, dnuy])\n%s\n" % sensi_matrix)
            return new_lattice

    def __sensi_matrix(self, components_list, target_values):
        """return the sensitive matrix to adjust components"""
        rela_matrix = np.zeros([len(target_values), len(components_list)])
        j = 0
        adjust_precision = 1e-4
        for comp in components_list:
            i = 0
            temp_lattice = Lattice(self.line.adjust(comp, adjust_precision), self.step, self.periods_number,
                                   self.coupl)
            for key in target_values.keys():
                rela_matrix[i, j] = temp_lattice.get_data(key) - self.get_data(key)
                i += 1
            j += 1
        return np.linalg.pinv(rela_matrix) * adjust_precision

    def __get_component(self, components_name_list):
        """according to the component_name_list generate the list of component objects"""

        components_list = []
        for comp in self.line.components:
            if comp.name in components_name_list:
                components_list.append(comp)
        return components_list

    def matching(self, components_name_list: list, target_values: dict, max_iter_times: int):
        """adjust the components in the list, and match lattice global parameters
        or beam parameters at the end of the lattice.

        you may need to set the limitation of components' parameter, para_max & para_min.
        for drift and dipole, the length will be adjusted.
        for quadrupole, the k1 will be adjusted.

        :param components_name_list: list of components, str
        :param target_values: { parameter name: value }
        :param max_iter_times: int
        """

        # TODO: 可以考虑独立成class, add different strategy

        assert len(components_name_list) >= len(target_values)
        assert isinstance(target_values, dict)
        components_list = self.__get_component(components_name_list)
        delta_data = np.array([v - self.get_data(k) for k, v in target_values.items()])
        sensi_matrix = self.__sensi_matrix(components_list, target_values)
        comps_changed = sensi_matrix.dot(delta_data)
        new_list = []
        i = 0
        # generate adjusted line and compare the value
        for comp in self.line.components:
            if comp in components_list:
                new_comp = comp.rela_adjust(1 + comps_changed[i])
                i += 1
            else:
                new_comp = copy.deepcopy(comp)
            new_list.append(new_comp)
        new_line = Line(new_list)
        new_lattice = Lattice(new_line, self.step, self.periods_number, self.coupl)
        if np.max(abs(np.array([v - new_lattice.get_data(k) for k, v in target_values.items()]))) < 1e-3:
            print("\n迭代完成\n剩余次数 %s, 迭代次数自己算\n" % (max_iter_times - 1))
            print(new_lattice.__get_component(components_name_list))
            return new_lattice
        elif max_iter_times > 1:
            local_values = [new_lattice.get_data(k) for k in target_values.keys()]
            print('\ncurrent values:\n' + str(local_values) + '\niteration continues...\n')
            return new_lattice.matching(components_name_list, target_values, max_iter_times - 1)
        else:
            print(new_line)
            print(new_lattice)
            raise Exception('迭代次数内未找到满足精度的值')

    def get_data(self, parameter: str):
        """receive parameter name as str, and return the corresponding object"""
        if parameter == 'nu_x' or parameter == 'nux':
            return self.nux
        elif parameter == 'nu_y' or parameter == 'nuy':
            return self.nuy
        elif parameter == 'betax' or parameter == 'beta_x' or parameter == 'xbeta':
            return self.elements[-1].beta_x
        elif parameter == 'betay' or parameter == 'beta_y' or parameter == 'ybeta':
            return self.elements[-1].beta_y
        elif parameter == 'eta':
            return self.elements[-1].eta
        elif parameter == 'etap':
            return self.elements[-1].etap
        elif parameter == 'xi_x':
            return self.xi_x
        elif parameter == 'xi_y':
            return self.xi_y
        else:
            raise UnfinishedWork(parameter)

    def __str__(self):
        val = ""
        val += ("nux =       " + str(self.elements[-1].nux * self.periods_number))
        val += ("\nnuy =       " + str(self.elements[-1].nuy * self.periods_number))
        val += ("\ncurl_H =    " + str(self.elements[-1].curl_H))
        val += ("\nI1 =        " + str(self.I1))
        val += ("\nI2 =        " + str(self.I2))
        val += ("\nI3 =        " + str(self.I3))
        val += ("\nI4 =        " + str(self.I4))
        val += ("\nI5 =        " + str(self.I5))
        val += ("\nJs =        " + str(self.Js))
        val += ("\nJx =        " + str(self.Jx))
        val += ("\nJy =        " + str(self.Jy))
        val += ("\nenergy =    " + str(Particle.energy) + "MeV")
        val += ("\ngamma =     " + str(Particle.gamma))
        val += ("\nsigma_e =   " + str(self.sigma_e))
        val += ("\nemittance = " + str(self.emittance * 1e9) + " nm*rad")
        val += ("\nLength =    " + str(self.line.length * self.periods_number) + " m")
        val += ("\nU0 =        " + str(self.U0 * 1000) + "  keV")
        val += ("\nTperiod =   " + str(self.Tperiod * 1e9) + " nsec")
        val += ("\nalpha =     " + str(self.alpha))
        val += ("\neta_p =     " + str(self.etap))
        val += ("\ntau0 =      " + str(self.tau0 * 1e3) + " msec")
        val += ("\ntau_e =     " + str(self.tau_s * 1e3) + " msec")
        val += ("\ntau_x =     " + str(self.tau_x * 1e3) + " msec")
        val += ("\ntau_y =     " + str(self.tau_y * 1e3) + " msec")
        val += ("\nxi_x =     " + str(self.xi_x))
        val += ("\nxi_y =     " + str(self.xi_y))
        if self.sigma_z is not None:
            val += ("\nsigma_z =   " + str(self.sigma_z))
        return val

    def save_plot_data(self):
        file1 = open("plot_data.txt", 'w')
        file1.write("s beta_x alpha_x gamma_x dmux mux beta_y alpha_y gamma_y dmuy muy eta eta'")
        for ele in self.elements:
            file1.write("\n%s %s %s %s %s %s %s %s %s %s %s %s %s" %
                        (ele.z_axis, ele.beta_x, ele.alpha_x, ele.gamma_x, ele.dmux, ele.mux, ele.beta_y, ele.alpha_y,
                         ele.gamma_y, ele.dmuy, ele.muy, ele.eta, ele.etap))
        file1.close()
        return 0


class PlotLattice(object):
    """View data quickly, if you want to analyse data, it's recommended to save_plot_data"""

    def __init__(self, lattice, parameters=None):
        assert lattice.elements
        self.paras = parameters
        self.lattice = lattice
        if parameters is not None:
            self.quick_view(self.paras)

    def quick_view(self, parameters):
        x = get_col(self.lattice.elements, 's')
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        x2, magnet_type = self.magnet_data()
        ax1.fill(x2, magnet_type, linestyle=':', color='#dddddd')
        plt.yticks([])
        ax2 = ax1.twinx()
        if isinstance(parameters, str):
            y = get_col(self.lattice.elements, parameters)
            ax2.plot(x, y, label=parameters)
        elif isinstance(parameters, list):
            for para in parameters:
                y = get_col(self.lattice.elements, para)
                ax2.plot(x, y, label=para)
        else:
            raise Exception('parameters should be str or list of str')
        ax2.legend()
        plt.show()

    def magnet_data(self):
        s = 0
        magenet_type = []
        z_axis = []
        for comp in self.lattice.line.components:
            magenet_type.append(comp.symbol)
            z_axis.append(s)
            s += comp.length
            z_axis.append(s)
            magenet_type.append(comp.symbol)
        return z_axis, magenet_type


def get_col(elements, parameter, *line_length):
    def __col_s():
        col_data = []
        for ele in elements:
            col_data.append(ele.z_axis)
        return col_data

    def __col_period_s():
        def ___generate__period_s(ss):
            if ss >= line_length[0]:
                ss = round(ss - line_length[0], LENGTH_PRECISION)
                return ___generate__period_s(ss)
            else:
                return ss

        col_data = []
        for ele in elements:
            col_data.append(___generate__period_s(ele.z_axis))
        return col_data

    def __col_x():
        col_data = []
        for ele in elements:
            col_data.append(ele.particle[0])
        return col_data

    def __col_xp():
        col_data = []
        for ele in elements:
            col_data.append(ele.particle[1])
        return col_data

    def __col_y():
        col_data = []
        for ele in elements:
            col_data.append(ele.particle[2])
        return col_data

    def __col_yp():
        col_data = []
        for ele in elements:
            col_data.append(ele.particle[3])
        return col_data

    def __col_z():
        col_data = []
        for ele in elements:
            col_data.append(ele.particle[4])
        return col_data

    def __col_delta():
        col_data = []
        for ele in elements:
            col_data.append(ele.particle[5])
        return col_data

    def __col_curl_H():
        col_data = []
        for ele in elements:
            col_data.append(ele.curl_H)
        return col_data

    def __col_beta_x():
        col_data = []
        for ele in elements:
            col_data.append(ele.beta_x)
        return col_data

    def __col_beta_y():
        col_data = []
        for ele in elements:
            col_data.append(ele.beta_y)
        return col_data

    def __col_eta():
        col_data = []
        for ele in elements:
            col_data.append(ele.eta)
        return col_data

    def __col_gamma_x():
        col_data = []
        for ele in elements:
            col_data.append(ele.gamma_x)
        return col_data

    def __col_gamma_y():
        col_data = []
        for ele in elements:
            col_data.append(ele.gamma_y)
        return col_data

    def __col_alpha_x():
        col_data = []
        for ele in elements:
            col_data.append(ele.alpha_x)
        return col_data

    def __col_alpha_y():
        col_data = []
        for ele in elements:
            col_data.append(ele.alpha_y)
        return col_data

    if parameter == 's':
        return __col_s()
    elif parameter == 'period_s':
        return __col_period_s()
    elif parameter == 'x':
        return __col_x()
    elif parameter == 'xp':
        return __col_xp()
    elif parameter == 'y':
        return __col_y()
    elif parameter == 'yp':
        return __col_yp()
    elif parameter == 'z':
        return __col_z()
    elif parameter == 'delta':
        return __col_delta()
    elif parameter == 'curl_H':
        return __col_curl_H()
    elif parameter == 'beta_x':
        return __col_beta_x()
    elif parameter == 'beta_y':
        return __col_beta_y()
    elif parameter == 'eta':
        return __col_eta()
    elif parameter == 'gamma_x':
        return __col_gamma_x()
    elif parameter == 'gamma_y':
        return __col_gamma_y()
    elif parameter == 'alpha_x':
        return __col_alpha_x()
    elif parameter == 'alpha_y':
        return __col_alpha_y()
    else:
        print('\n' + str(parameter) + " hasn't been defined")
        raise UnfinishedWork(str(parameter))


class PlotTrack(object):
    def __init__(self, track):
        self.track = track

    def plot_along(self, parameters, plot_type='lattice'):
        if plot_type == 'lattice':
            for elements in self.track.elements:
                s = get_col(elements, 'period_s', self.track.line.length)
                x = get_col(elements, parameters)
                plt.plot(s, x)
            plt.xlabel('s')
            plt.ylabel(str(parameters))
            plt.show()
        elif plot_type == 'long_line':
            s = []
            data = []
            for elements in self.track.elements:
                s += get_col(elements, 's')
                data += get_col(elements, parameters)
            plt.plot(s, data)
            plt.xlabel('s')
            plt.ylabel(str(parameters))
            plt.show()
        else:
            raise ValueError('plot_type must be lattice or long_line')

    def plot_phase_space(self, parameters):
        x = []
        y = []
        for elements in self.track.elements:
            x += get_col(elements, parameters[0])
            y += get_col(elements, parameters[1])
        plt.scatter(x, y, marker='.', s=1)
        plt.xlabel(str(parameters[0]))
        plt.ylabel(str(parameters[1]))
        plt.show()


class Track(object):
    """tracking, may need another Step

    Track concerns about the particles position, not the integral quantities, so the step of Drift don't need to be
    small. But other magnets, especially nonlinear components like Sextupole, should be slided to get more accurate
    result."""

    def __init__(self, line, step: Step):
        if isinstance(line, Line):
            self.line = line
            self.periods = 1
        elif isinstance(line, Lattice):
            self.line = line.line
            self.periods = line.periods_number
            # self.rf_cavity = line.rf_cavity
            # self.U0 = line.U0
        else:
            raise TypeError('line should be Line or Lattice')
        self.marks = [0.0]
        self.turns = None
        self.elements = None
        self.step = step
        self.initial_s = 0
        self.record = []
        self.line_elements = self.slice_line()

    def slice_line(self):
        elements = []
        ele = Element()
        ele.z_axis = 0
        magnet, drop_data = self.line.find_component(0)
        ele.symbol = magnet.symbol
        ele.theta_e = 0
        ele.get_magnet_data(magnet)
        for matrix, zi, is_edge in self.line.step_matrix(0, self.step, self.line.length):
            ele.length = round(zi - ele.z_axis, LENGTH_PRECISION)  # between z_i and z_(i-1)
            ele.matrix = matrix.matrix
            if is_edge:
                if 200 <= ele.symbol < 300:
                    ele.symbol = 242
                    last_magnet, drop_data = self.line.find_component(round(ele.z_axis, LENGTH_PRECISION))
                    ele.theta_e = last_magnet.outlet.theta_e
            elements.append(copy.deepcopy(ele))
            ele.z_axis = round(zi, LENGTH_PRECISION)
            magnet, drop_data = self.line.find_component(zi)
            ele.get_magnet_data(magnet)
            if is_edge and 200 <= magnet.symbol < 300:
                ele.symbol = 241
                ele.theta_e = magnet.inlet.theta_e
            else:
                ele.symbol = magnet.symbol
                ele.theta_e = 0
        ele.length = 0
        ele.matrix = np.identity(6)
        elements.append(copy.deepcopy(ele))
        return elements

    def set_initial_s(self, initial_s):
        raise UnfinishedWork('initial s is 0.0')

    def set_marks(self, s):
        """to set marks along the ring, and data at mark will be output"""
        if s != 0:
            print('\n目前只支持记录入口处数据，标记将被改到0.0处')
        print('\nset marks at s=0.0 successfully!')
        self.marks = [0.0]

    def set_step(self, step: Step):
        self.step = step

    def do_track(self, particle, turns, record_data='simplified', plot=1):
        """tracking particle for n turns

        :param particle: particle's 6D position
        :param turns: n turns
        :param record_data: 'detailed' or 'simplified'
        :param plot: default 1

        :return the final 6D position"""
        self.turns = turns
        if record_data == 'detailed':
            elements = []
            for i in range(turns):
                for j in range(self.periods):
                    current_elements = self.__track_detailed_data(particle, (i * self.periods + j) * self.line.length,
                                                                  j * self.line.length)
                    particle = current_elements[-1].particle
                    current_elements.pop()
                    elements.append(current_elements)
                # the eles at the end of the last list and at the beginning of the new list share the same values,
                # but the length of the previous one is 0.
            self.elements = elements
        elif record_data == 'simplified':
            self.record = []
            for i in range(turns):
                for j in range(self.periods):
                    try:
                        particle = self.__track_simplify_data(particle, (i * self.periods + j) * self.line.length,
                                                              j * self.line.length)
                    except ParticleLost:
                        print('record ' + str(len(self.record)) + ' data')
                        raise ParticleLost
                print('finished ' + str(i + 1) + ' turns')
            print(str(len(self.record)) + ' data recorded')
        else:
            raise TypeError
        if plot == 1:
            self.fft_phase()

    def __track_simplify_data(self, particle, z0, z_in_ring):
        """the process of calculation is same as detailed data, but only record the final data and the data of at
        marked position"""
        for line_ele in self.line_elements:
            line_ele.particle = particle
            try:
                particle = line_ele.next_particle()
            except ParticleLost:
                raise ParticleLost(line_ele.z_axis + z0, z_in_ring)
            if round(line_ele.z_axis + z_in_ring, 15) in self.marks:
                self.record.append(line_ele.particle)
        return particle

    def __track_detailed_data(self, particle, z0, z_in_ring):
        """generate series of elements, recording all the data along the line"""
        elements = []
        ele = Element()
        ele.particle = particle
        for line_ele in self.line_elements:
            ele.z_axis = round(line_ele.z_axis + z0, LENGTH_PRECISION)
            ele.length = line_ele.length
            line_ele.particle = ele.particle
            elements.append(copy.deepcopy(ele))
            try:
                ele.particle = line_ele.next_particle()
            except ParticleLost:
                print('particle lost at ' + str(ele.z_axis))
                return elements
            if round(line_ele.z_axis + z_in_ring, 15) in self.marks:
                self.record.append(line_ele.particle)
        return elements

    def fft_phase(self):
        """plot phase space and do fft analysis"""
        x = []
        px = []
        y = []
        py = []
        for particle in self.record:
            x.append(particle[0])
            px.append(particle[1])
            y.append(particle[2])
            py.append(particle[3])
        fft_tune = fftpack.fft(y)
        k1 = [i * 1 / self.turns for i in range(int(self.turns / 2))]
        abs_y = np.abs(fft_tune[range(int(self.turns / 2))])
        normalized_y = abs_y / self.turns
        y_peak = signal.find_peaks(normalized_y)
        sorted_y_peak = self.__resort_peak_index(y_peak[0], normalized_y)
        y_peak_position = [k1[i] for i in y_peak[0]]
        sorted_y_peak_position = [k1[i] for i in sorted_y_peak]
        print('y tune peak is at \n' + str(sorted_y_peak_position))
        fft_tune_x = fftpack.fft(x)
        abs_x = np.abs(fft_tune_x[range(int(self.turns / 2))])
        normalized_x = abs_x / self.turns
        x_peak = signal.find_peaks(normalized_x)
        sorted_x_peak = self.__resort_peak_index(x_peak[0], normalized_x)
        x_peak_position = [k1[i] for i in x_peak[0]]
        sorted_x_peak_position = [k1[i] for i in sorted_x_peak]
        print('x tune peak is at \n' + str(sorted_x_peak_position))
        plt.figure()
        plt.subplot(224)
        plt.plot(k1, normalized_y)
        plt.scatter(y_peak_position, normalized_y[y_peak[0]], s=5, color='red')
        plt.title('fft of y ')
        plt.subplot(223)
        plt.plot(k1, normalized_x)
        plt.scatter(x_peak_position, normalized_x[x_peak[0]], s=5, color='red')
        plt.title('fft of x')
        plt.subplot(221)
        xe3 = [i * 1000 for i in x]
        pxe3 = [i * 1000 for i in px]
        plt.scatter(xe3, pxe3, s=1)
        plt.xlabel('x [mm]')
        plt.ylabel('px [mrad]')
        plt.title('phase space x')
        plt.subplot(222)
        ye3 = [i * 1000 for i in y]
        pye3 = [i * 1000 for i in py]
        plt.scatter(ye3, pye3, s=1)
        plt.ylabel('py [mrad]')
        plt.xlabel('y [mm]')
        plt.title('phase space y')
        plt.show()

    def __resort_peak_index(self, peak, value):
        peak_value = [value[i] for i in peak]
        peak_value.sort(reverse=True)
        resort_index = [np.where(value == i)[0][0] for i in peak_value]
        return resort_index


    def save_record(self):
        file1 = open("record_particle.txt", 'w')
        file1.write("x px y py ")
        for particle in self.record:
            file1.write("\n%s %s %s %s" % (particle[0], particle[1], particle[2], particle[3]))
        file1.close()
        return 0

    def save_plot_data(self):
        file1 = open("particle_data.txt", 'w')
        file1.write("s x px y py z delta")
        for elements in self.elements:
            for ele in elements:
                file1.write("\n%s %s %s %s %s %s %s" % (ele.z_axis, ele.particle[0], ele.particle[1], ele.particle[2],
                                                        ele.particle[3], ele.particle[4], ele.particle[5]))
        file1.close()
        return 0
