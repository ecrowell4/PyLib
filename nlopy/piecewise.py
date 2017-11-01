# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:15:04 2017

@author: Owner
"""

# Define position space
L = 1
l = L / 8
N = 75
dx = l / N

x0 = np.arange(N)*dx
x = np.arange(8*N)*dx

V = Vcoeffs[0]*x0
inter = np.zeros(1)
xi = x0
# Construct potential:
for i in range(1,8):
    xi = xi + i/8
    bi = (Vcoeffs[i-1] - Vcoeffs[i]) * xi[0] + inter[i-1]
    inter = np.append(inter, bi)
    V = np.append(V, Vcoeffs[i]*(xi) + bi)
    
plt.figure()
plt.plot(x, V)
plt.show()

#==============================================================================
# Taylor Series Method
#==============================================================================
L = 1
N = 1000
dx = L / N
x = np.arange(N) * dx

num = 10
coeffs1 = np.random.randn(2*num)
coeffs2 = 10*np.random.randn(2*num)
f = np.zeros(len(x))
for l in range(num):
    f += (coeffs1[l] / 1) * (x - coeffs1[l+num])**l + coeffs2[l+num]*np.cos(2*l*np.pi*x/L) + coeffs2[l+num]*np.sin(2*l*np.pi*x/L)

plt.figure()
plt.plot(x, f)
plt.show()