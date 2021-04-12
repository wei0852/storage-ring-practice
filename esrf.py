from simplestoragering import *

Particle.set_energy(6000)
D1 = Drift('D1', 3.169600)
D2 = Drift('D2', 0.300000)
D3 = Drift('D2', 0.350000)
D4 = Drift('D3', 1.072250)
D5 = Drift('D4', 1.071750)
D6 = Drift('D5', 0.380000)
D7 = Drift('D6', 0.450000)

QD1 = Quadrupole('QD1', 0.400000, -0.319693)
QF1 = Quadrupole('QF1', 0.900000, 0.529010)
QD2 = Quadrupole('QD2', 0.500000, -0.524120)
QD3 = Quadrupole('QD3', 0.400000, -0.693040)
QF2 = Quadrupole('QF2', 0.500000, 0.759003)
QD4 = Quadrupole('QD4', 0.500000, -0.770779)
QF3 = Quadrupole('QF3', 0.900000, 0.819497)
QD5 = Quadrupole('QD5', 0.400000, -0.547108)

B1 = Dipole('B1', 2.157280, 5.289780 * pi / 180, 2.812500 * pi / 180, 2.477300 * pi / 180)
B2 = Dipole('B2', 0.292710, 0.335180 * pi / 180, -2.477300 * pi / 180, 2.812500 * pi / 180)

SF = Sextupole('SF', 0.100000, 31.03000)
SD = Sextupole('SD', 0.100000, -17.75000)
S1 = Sextupole('S1', 0.100000, 20.00000)
S2 = Sextupole('S2', 0.100000, -22.60000)
S3 = Sextupole('S3', 0.100000, -10.80000)
S4 = Sextupole('S4', 0.100000, 18.10000)

MHB = [QD2, D3, S2, D3, QF1, D2, S1, D2, QD1, D1]
DBA = [D4, B1, B2, D5, QD3, D2, SD, D6, QF2, D7, SF, D7, QF2, D6, SD, D2, QD3, D5, B2, B1, D4]
MLB = [QD4, D2, S3, D3, QF3, D2, S4, D2, QD5, D1]
MHB.reverse()
segment1 = MHB + DBA + MLB
segment2 = copy.deepcopy(segment1)
segment2.reverse()
segment = segment1 + segment2
line = Line(segment)
print(line.length * 16)
step = Step({0: 0.001})
lattice = Lattice(line, step, 16, 0.00)
print(lattice)
PlotLattice(lattice, ['beta_x', 'beta_y'])

print(lattice.tune_adjust(36.81, 11.34))

