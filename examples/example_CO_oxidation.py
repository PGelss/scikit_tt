#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scikit_tt.tensor_train as tt
import scikit_tt.models as mdl
import scikit_tt.solvers.EVP as evp
import scikit_tt.solvers.ODE as ODE
import scikit_tt.tools as tls
import matplotlib.pyplot as plt


def turn_over_frequency(distribution):
    tt_left = [None] * distribution.order
    tt_right = [None] * distribution.order
    for i in range(distribution.order):
        tt_left[i] = tt.TT.ones([1] * distribution.order, distribution.row_dims)
        tt_right[i] = tt.TT.ones([1] * distribution.order, distribution.row_dims)
        tt_left[i].cores[i][0, 0, :, 0] = [0, 0, 1]
        tt_right[i].cores[i][0, 0, :, 0] = [0, 0, 1]
        if i > 0:
            tt_left[i].cores[i - 1][0, 0, :, 0] = [0, 1, 0]
        else:
            tt_left[i].cores[-1][0, 0, :, 0] = [0, 1, 0]
        if i < distribution.order - 1:
            tt_right[i].cores[i + 1][0, 0, :, 0] = [0, 1, 0]
        else:
            tt_right[i].cores[0][0, 0, :, 0] = [0, 1, 0]
    tof = 0
    k_de_CO2 = 1.7e5
    for i in range(distribution.order):
        tof = tof + (k_de_CO2 / distribution.order) * (tt_left[i] @ distribution).element([0]*distribution.order*2) + \
              (k_de_CO2 / distribution.order) * (tt_right[i] @ distribution).element([0]*distribution.order*2)
    return tof

TOF = []
p_CO_exp = [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2]


print('\n----------------------------------------------------------')
print('p_CO in atm    Method    TT ranks    Closeness    CPU time')
print('----------------------------------------------------------')

R = [3,3,4,7,8,8,10,10]
for i in range(len(R)):
    operator = mdl.co_oxidation(20, 10 ** (8 + p_CO_exp[i]))
    initial_guess = tt.TT.ones(operator.row_dims,[1]*operator.order, ranks=R[i]).ortho_right()
    with tls.Timer() as time:
        solution, eigenvalues = evp.als(tt.TT.eye(operator.row_dims)+operator, initial_guess, number_ev=1, repeats=20)
        print(eigenvalues)
        solution = (1/solution.norm(p=1))*solution
    TOF.append(turn_over_frequency(solution).copy())
    print('10^'+(p_CO_exp[i]>=0)*'+'+str("%.1f" % p_CO_exp[i])+'         EVP         '+(R[i]<10)*' '+str(R[i])+'        '+str("%.2e" % (operator@solution).norm())+(time.elapsed<10)*' '+'      '+str("%.2f" % time.elapsed) + 's')

# R = [14, 15, 17, 18]
# for i in range(3,len(R)):
#     operator = mdl.CO_oxidation(20, 10 ** (8 + p_CO_exp[8+i]))
#     initial_value = tt.TT.zeros(operator.row_dims, [1] * operator.order)
#     for j in range(initial_value.order):
#         initial_value.cores[j][0,1,0,0] = 1
#     initial_guess = tt.TT.ones(operator.row_dims, [1] * operator.order, ranks=R[i]).ortho_right()
#     with tls.Timer() as time:
#         solution = ODE.adaptive_als(operator, initial_value, initial_guess, 10**-10, 10, repeats = 5)
#     TOF.append(turn_over_frequency(solution[-1]).copy())
#     print('10^+' + str("%.1f" % p_CO_exp[i+8]) + '         IEM         ' + (R[i] < 10) * ' ' + str(
#         R[i]) + '        ' + str("%.2e" % (operator @ solution[-1]).norm()) + (time.elapsed < 10) * ' ' + '      ' + str(
#         "%.2f" % time.elapsed) + 's')

print('------------------------------------------------')

# plt.plot(p_CO_exp[:len(R)],TOF)
# plt.yscale('log')
# plt.show()
