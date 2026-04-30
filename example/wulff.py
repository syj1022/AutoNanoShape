import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
import pandas as pd
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core.surface import Lattice, SlabGenerator, Structure, generate_all_slabs

"""
This script demonstrates an example of plotting of MLIP-predicted CaO Wulff shape at 300 K
and 1 atm water partial pressure.
"""

std_mu_data = {
    "T": [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    "μ˚_H2O": [-0.295, -0.388, -0.484, -0.584, -0.686, -0.79, -0.896, -1.004, -1.113, -1.225, -1.337, -1.451, -1.567, -1.683, -1.801, -1.92, -2.04]
}

df = pd.DataFrame(std_mu_data)

e_h2o = -496.2706
e_o = -460.8683
bulk = -5960.7292
e_stoich = [2, 4, 4, 4, 4]
g_stoich = [2, 2, 4, 0, 4]
factor = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

def get_mu_water(T, P):
    mu_std_H2O = interp1d(df["T"], df["μ˚_H2O"], kind='linear', fill_value="extrapolate")
    mu_H2O = 51894796115390900000 * T * np.log(P)/6.0221408e23 + mu_std_H2O(T)
    return mu_H2O

def bare_energy(raw_total, bulk, layer, A, i, correction):
    return (raw_total[0]-bulk*layer)/2/A+correction*2*e_o/A

def H_energy(raw_total, raw_bare_total, layer, A, T, P, i):
    SE_half = (raw_total[0] - raw_bare_total[0] - e_stoich[0]*(e_h2o-e_o) - g_stoich[0]*get_mu_water(T,P))/2/A
    SE_full = (raw_total[1] - raw_bare_total[1] - e_stoich[0]*(e_h2o-e_o)*factor[i] - g_stoich[0]*get_mu_water(T,P)*factor[i])/2/A
    SE_single = (raw_total[2] - raw_bare_total[2] - e_stoich[0]*(e_h2o-e_o)/2 - g_stoich[0]*get_mu_water(T,P)/2)/2/A
    return np.min([SE_half, SE_full, SE_single])

def OH_energy(raw_total, raw_bare_total, layer, A, T, P, i):
    SE_half = (raw_total[0] - raw_bare_total[0] - e_stoich[1]*(e_o+0.5*(e_h2o-e_o)) - g_stoich[1]*get_mu_water(T,P))/2/A
    SE_full = (raw_total[1] - raw_bare_total[1] - e_stoich[1]*(e_o+0.5*(e_h2o-e_o))*factor[i] - g_stoich[1]*get_mu_water(T,P)*factor[i])/2/A
    SE_single = (raw_total[2] - raw_bare_total[2] - e_stoich[1]*(e_o+0.5*(e_h2o-e_o))/2 - g_stoich[1]*get_mu_water(T,P)/2)/2/A
    return np.min([SE_half, SE_full, SE_single])

def H_OH_energy(raw_total, raw_bare_total, layer, A, T, P, i):
    SE_half = (raw_total[0] - raw_bare_total[0] - e_stoich[2]*e_h2o - g_stoich[2]*get_mu_water(T,P))/2/A
    SE_full = (raw_total[1] - raw_bare_total[1] - e_stoich[2]*e_h2o*factor[i] - g_stoich[2]*get_mu_water(T,P)*factor[i])/2/A
    SE_single = (raw_total[2] - raw_bare_total[2] - e_stoich[2]*e_h2o/2 - g_stoich[2]*get_mu_water(T,P)/2)/2/A
    return np.min([SE_half, SE_full, SE_single])

def O_energy(raw_total, raw_bare_total, layer, A, T, P, i):
    SE_half = (raw_total[0] - raw_bare_total[0] - e_stoich[3]*e_o)/2/A
    SE_full = (raw_total[1] - raw_bare_total[1] - e_stoich[3]*e_o*factor[i])/2/A
    SE_single = (raw_total[2] - raw_bare_total[2] - e_stoich[3]*e_o/2)/2/A
    return np.min([SE_half, SE_full, SE_single])

def water_energy(raw_total, raw_bare_total, layer, A, T, P, i):
    SE_half = (raw_total[0] - raw_bare_total[0] - e_stoich[4]*e_h2o - g_stoich[4]*get_mu_water(T,P))/2/A
    SE_full = (raw_total[1] - raw_bare_total[1] - e_stoich[4]*e_h2o*factor[i] - g_stoich[4]*get_mu_water(T,P)*factor[i])/2/A
    SE_single = (raw_total[2] - raw_bare_total[2] - e_stoich[4]*e_h2o/2 - g_stoich[4]*get_mu_water(T,P)/2)/2/A
    return np.min([SE_half, SE_full, SE_single])

### replace by your data
def SE(T, P):
    SE = []
    cell = [[3.5, 23.5051], #100
            [4.5, 33.2412], #110
            [4, 40.7963], #111*
            [5, 52.6678], #210
            [5, 57.6946], #211*
            [6, 70.6612], #221*
            [8, 74.4834], #310
            [8, 78.1189], #311-Ca
            [9, 78.1189], #311-O
            [10, 84.9242], #320
            [10, 88.1300], #321
            [10, 97.1145], #322
            [10, 102.6683], #331-Ca
            [9, 102.6683], #331-O
            [10, 110.4768]] #332

    data = [[-20860.29267, -20931.75843, -22772.47697, -22848.06667, -22705.07086, -22848.61753],
            [-20860.29267, -20931.75843, -22772.47697, -22848.06667, -22705.07086, -22848.61753],
            [-20860.29267, -20896.01962, -21816.81394, -21854.37692, -21777.34559, -21853.2852],
            [-26817.35633, -26889.90714, -28732.72534, 0, -28661.47001, -28804.46501],
            [-26817.35633, -26889.90714, -28732.72534, 0, -28661.47001, -28804.46501],
            [-26817.35633, -26854.05057, -27774.53858, -27814.45023, -27734.72129, -27811.78703],
            [-21967.92369, 0, -23908.4226, 0, -23832.02164, 0],
            [-21967.92369, 0, 0, 0, -25674.17775, -25954.7374],
            [-21967.92369, 0, -22948.55792, 0, -22910.00544, -22982.8312],
            [-29795.61731, -29869.25275, -31711.98385, -31786.08346, -31640.78349, -31784.01127],
            [-29795.61731, -29941.27415, -33627.78777, -33777.612, -33487.94069, -33773.41292],
            [-29795.61731, -29831.99033, -30753.61209, -30791.17804, -30718.71225, -30789.60525],
            [-31639.2198, -31712.54683, -33554.05503, -33627.49385, -33482.76815, -33625.57732],
            [-31639.2198, -31790.06666, -35470.7931, -35618.71065, -35327.64822, 0],
            [-31639.2198, -31674.00579, -32597.98583, -32633.70397, -32558.14438, -32636.36569],
            [-37592.45366, -37669.34703, -39511.8419, -39585.66887, -39439.1799, -39585.2983],
            [-37592.45366, -37747.09799, -41427.8272, -41575.11895, -41282.73606, 0],
            [-37592.45366, -37631.21802, -38554.18243, -38590.72644, -38515.96107, -38589.20284],
            [-47675.29552, -47748.74834, -49590.31737, -49665.743, -49521.65334, -49662.40719],
            [-47675.29552, -47820.07077, -51504.39844, -51653.26089, -51355.06376, -51651.00869],
            [-47675.29552, -47711.82789, -48632.956, -48669.1215, -48598.65639, -48668.91312],
            [-45821.84667, -45895.05038, -47746.33365, -47818.55806, -47670.50434, -47813.48231],
            [-45821.84667, -45967.04444, -49664.07753, -49803.93399, -49519.26981, -49805.52641],
            [-45821.84667, -45859.30483, -46784.80185, -46820.87784, -46746.73376, -46820.95606],
            [-55469.87909, -55546.61931, -57385.44938, -57463.4854, -57315.00581, -57466.86144],
            [-55469.87909, -55623.47271, -59302.36774, -59454.10435, -59159.35887, -59453.00333],
            [-55469.87909, -55508.68505, -56428.78683, -56467.44094, -56393.82489, -56466.85415],
            [-59593.06087, -59666.19914, -61510.5554, -61583.41427, -61440.74032, -61583.76967],
            [-59593.06087, -59375.93208, -63424.89254, -63575.91642, -63283.29999, -63572.42012],
            [-59593.06087, -59629.55987, -60551.88186, -60589.35279, -60516.0977, -60588.15202],
            [-59586.42901, -59662.26845, -61506.74963, -61580.929, -61436.04219, -61579.81631],
            [-59586.42901, -59734.92594, -63426.83009, -63570.91579, -63280.59882, -63569.43534],
            [-59586.42901, -59624.76331, -60546.1528, -60583.77085, -60512.3968, -60582.75711],
            [-59582.09551, -59653.34618, -61504.78239, -61580.11912, -61429.20248, 0],
            [-59582.09551, -59728.81443, -63422.53211, -63570.50762, -63276.48513, -63564.67586],
            [-59582.09551, -59614.5593, -60544.07521, -60581.62896, -60506.18022, -60577.22273],
            [-57736.52602, -57810.11095, -59660.00321, -59732.45355, -59581.78779, -59729.75545],
            [-57736.52602, -57884.62856, -61583.94722, -61720.78563, -61430.11307, -61723.05872],
            [-57736.52602, -57775.58921, -58699.47216, -58734.39835, -58661.38843, -58732.26862],
            [-55469.87909, -55546.61931, -57385.44938, -57463.4854, -57315.00581, -57466.86144],
            [-55469.87909, -55623.47271, -59302.36774, -59454.10435, -59159.35887, -59453.00333],
            [-55469.87909, -55508.68505, -56428.78683, -56467.44094, -56393.82489, -56466.85415],
            [-59573.45415, -59656.15752, -61498.93061, -61575.95246, -61421.69348, -61572.20209],
            [-59573.45415, -59723.58139, -63421.04051, -63566.24866, -63271.20294, -63558.22517],
            [-59573.45415, -59613.79076, -60536.72455, -60574.84658, -60501.07682, -60571.75294]]

    for i in range(15):
        if i in [2, 7, 12]:
            correction = 1
        elif i in [4, 5, 8, 13]:
            correction = -1
        else:
            correction = 0
        SE.append(bare_energy([row[0] for row in data[3*i:3*i+3]], bulk, cell[i][-2], cell[i][-1], i, correction)*16.022 + np.min([0,
            H_energy([row[1] for row in data[3*i:3*i+3]], [row[0] for row in data[3*i:3*i+3]], cell[i][-2], cell[i][-1], T, P, i),
            OH_energy([row[2] for row in data[3*i:3*i+3]], [row[0] for row in data[3*i:3*i+3]], cell[i][-2], cell[i][-1], T, P, i),
            H_OH_energy([row[3] for row in data[3*i:3*i+3]], [row[0] for row in data[3*i:3*i+3]], cell[i][-2], cell[i][-1], T, P, i),
            O_energy([row[4] for row in data[3*i:3*i+3]], [row[0] for row in data[3*i:3*i+3]], cell[i][-2], cell[i][-1], T, P, i),
            water_energy([row[5] for row in data[3*i:3*i+3]], [row[0] for row in data[3*i:3*i+3]], cell[i][-2], cell[i][-1], T, P, i)])*16.022)

    return SE



T = 300
P = 1e-3

surface_energies = {(1, 0, 0): SE(T, P)[0], (1, 1, 0): SE(T, P)[1], (1, 1, 1): SE(T, P)[2], (2, 1, 0): SE(T, P)[3], (2, 1, 1): SE(T, P)[4], (2, 2, 1): SE(T, P)[5], (3, 1, 0): SE(T, P)[6], (3, 1, 1): SE(T, P)[7], (3, 1, 1): SE(T, P)[8], (3, 2, 0): SE(T, P)[9], (3, 2, 1): SE(T, P)[10], (3, 2, 2): SE(T, P)[11], (3, 3, 1): SE(T, P)[12], (3, 3, 1): SE(T, P)[13], (3, 3, 2): SE(T, P)[14]}
print(surface_energies)

lattice = Lattice.cubic(4.853)
structure = Structure(lattice, ["Ca", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
wulffshape = WulffShape(structure.lattice, surface_energies, list(surface_energies.values()))

custom_colors = {(1, 0, 0): "orange", (1, 1, 0): "green", (1, 1, 1): "blue", (2, 1, 0): "red", (2, 1, 1): "pink", (2, 2, 1): "brown", (3, 1, 0): "purple", (3, 1, 1): "cyan", (3, 2, 0): "gray", (3, 2, 1): "orchid", (3, 2, 2): "darkblue", (3, 3, 1): "teal", (3, 3, 2): "gold"}

plt.figure(figsize=(10, 10), dpi=300)

ax = wulffshape.get_plot(custom_colors=custom_colors, direction=(1.00, 0.5, 0.5), aspect_ratio=(10, 10))

for collection in ax.collections:
    collection.set_edgecolor('black')
    collection.set_linewidth(3)

plt.show()
