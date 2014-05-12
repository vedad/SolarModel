import numpy as np
import sys
import time
import matplotlib.pyplot as plt

### Constants ###
#-------------------------------------------------------------#
# Physical constants
_M_U = 1.6605e-24		# Units of [g]						CGS
_C = 3e10				# Units of [cm s^-1]				CGS
_K = 1.38e-16			# Units of [erg K^-1]				CGS
_SIGMA = 5.67e-5		# Units of [erg cm^-2 s^-1 K^-4]	CGS
_G = 6.67e-8			# Units of [cm^3 g^-1 s^-2]			CGS
_A = 4 * _SIGMA / _C	#

# Abundancy of different elements
X = 0.7				# Hydrogen (ionised)
Y = 0.29			# Helium 4 (ionised)
Y_3 = 1e-10			# Helium 3 (ionised)
Z = 0.01			# Heavier elements than the above (ionised)
Z_7Li = 1e-12		# Lithium 7 (part of Z)
Z_7Be = 1e-12		# Beryllium 7 (part of Z)

# Average molecular weight
_MU = 1./(2*X + 3* Y/4. + Z/2.)
#-------------------------------------------------------------#



def opacity(T,rho):
	"""
	Reads opacities from a file. Finds the opacity value
	that most closely resembles the present values for
	T and R.
	Returns the opacity in units of [cm^2 g^-1].
	"""

	logT = []; logK = []
	inFile = open('opacity.txt', 'r')
	
	# Read the header file to store log(R)
	logR = np.asarray(inFile.readline().strip().split()[1:], dtype=np.float64)

	inFile.readline() # Skip the header

	# Adding log(T) and log(khappa) in separate lists
	for line in inFile:
		logT.append(line.strip().split()[0])
		logK.append(line.strip().split()[1:])

	inFile.close()
	# Converts the array to contain 64 bit floating point numbers
	logT = np.asarray(logT, dtype=np.float64)
	logK = np.asarray(logK, dtype=np.float64)

	R = rho / (T / 1e6)	# Definition of R given in the opacity file

	# Make two arrays that contain the difference in T and R from present values
	diffT = abs(10**(logT) - T)
	diffR = abs(10**(logR) - R)

	# Finds the index of the minimum difference values, so the most relevant kappa can be used.
	i = np.argmin(diffT)
	j = np.argmin(diffR)
	
	kappa = 10**(logK[i,j])
	return kappa


def energyGeneration(T, rho):
	"""
	Function to find the full energy generation per
	unit mass from the three PP chain reactions.
	"""

	ergs = 1.602e-6			# Conversion from MeV to ergs (CGS)
	NA_inv = 1./6.022e23	# Avogadro's constant inverse
	
	### Energy values (Q) ###

	# Energy values of nuclear reactions in PP I [MeV]
	Q_pp = (1.177 + 5.949) * ergs ; Q_3He3He = 12.86 * ergs

	# Energy values of nuclear reactions in PP II [MeV]
	Q_3He4He = 1.586 * ergs ; Q_e7Be = 0.049 * ergs ; Q_p7Li = 17.346 * ergs

	# Energy values of nuclear reactions in PP III [MeV]
	Q_p7Be= (0.137 + 8.367 + 2.995) * ergs

	### Number densities (n) ###
	n_p = X * rho / _M_U				# Hydrogen
	n_3He = Y_3 * rho / (3 * _M_U)		# Helium 3
	n_4He = Y * rho / (4 * _M_U)		# Helium 4
	n_7Be = Z_7Be * rho / (7 * _M_U)	# Beryllium 7
	n_e = n_p + 2 * n_4He				# Electron
	n_7Li = Z_7Li * rho / (7 * _M_U)	# Lithium 7

	### Reaction rates (lambda) ### Units of reactions per second per cubic cm. [cm^3 s^-1]
	T9 = T / (1e9)

	l_pp = (4.01e-15 * T9**(-2./3) * np.exp(-3.380 * T9**(-1./3)) * (1 + 0.123 * T9**(1./3)\
			+ 1.09 * T9**(2./3) + 0.938 * T9)) * NA_inv

	l_3He3He = (6.04e10 * T9**(-2./3) * np.exp(-12.276 * T9**(-1./3))\
			* (1 + 0.034 * T9**(1./3) - 0.522 * T9**(2./3) - 0.124 * T9 +\
			0.353 * T9**(4./3) + 0.213 * T9**(5./3)))*NA_inv

	a = 1 + 0.0495 * T9 # Defined for efficiency in l_3He4He
	l_3He4He = (5.61e6 * a**(-5./6) * T9**(-2./3) * np.exp(-12.826 * a**(1./3) * T9**(-1./3)))\
			* NA_inv
	
	# Check if temperature is below 1e6 for this reaction. Special case if true.
	if T < 1e6:
		l_e7Be = (1.51e-7 / n_e) * NA_inv
	else:
		l_e7Be = (1.34e-10 * T9**(-1./2) * (1 - 0.537 * T9**(1./3) + 3.86 * T9**(2./3)\
			+ 0.0027 * T9**(-1.) * np.exp(2.515e-3 * T9**(-1.)))) * NA_inv

	a = 1 + 0.759 * T9 # Defined for efficiency in l_p7Li
	l_p7Li = (1.096e9 * T9**(-2./3) * np.exp(-8.472 * T9**(-1./3)) - 4.830e8 * a**(-5./6)\
	* T9**(-2./3) * np.exp(-8.472 * a**(1./3) * T9**(-1./3))) * NA_inv

	l_p7Be = (3.11e5 * T9**(-2./3) * np.exp(-10.262 * T9**(-1./3))) * NA_inv

	### Rates per unit mass (r) ### 
	r_pp = l_pp * (n_p * n_p) / (rho * 2)
	r_3He3He = l_3He3He * (n_3He * n_3He) / (rho * 2)
	r_3He4He = l_3He4He * (n_3He * n_4He) / rho
	r_e7Be = l_e7Be * (n_7Be * n_e) / rho
	r_p7Li = l_p7Li * (n_p * n_7Li) / rho
	r_p7Be = l_p7Be * (n_p * n_7Be) / rho

	### Energy generation per unit mass from PP I, II and III ###
	PP1 = (Q_pp * r_pp) + (Q_3He3He * r_3He3He)
	PP2 = (Q_pp * r_pp) + (Q_3He4He * r_3He4He) + (Q_e7Be * r_e7Be) + (Q_p7Li * r_p7Li)
	PP3 = (Q_pp * r_pp) + (Q_3He4He * r_3He4He) + (Q_p7Be * r_p7Be)

	epsilon = (Q_pp * r_pp) + (Q_3He3He * r_3He3He) + (Q_3He4He * r_3He4He) + (Q_e7Be *
			r_e7Be) + (Q_p7Li * r_p7Li) + (Q_p7Be * r_p7Be)
	
	return epsilon, PP1, PP2, PP3

def total_flux(L,r):
	"""
	Calculates the total flux, from convective and radiative energy transfer.
	"""
	F = L/(4 * np.pi * r*r)
	return F

def convective_flux(F,F_R):
	"""
	Calcultes the flux from convective energy transfer.
	"""
	F_C = F - F_R
	return F_C

def radiative_flux(m,r,T,P,kappa,nabla):
	"""
	Calculates the flux from radiative energy transfer.
	"""
	F_R = 4 * _A * _C * _G * T*T*T*T * m / (3 * kappa * P * r*r) * nabla
	return F_R
	
def nabla_radiation(T,P,L,m,kappa):
	"""
	Calculates the temperature gradient considering
	radiative energy transfer only.
	"""
	return 3 * kappa * L * P / (64 * np.pi * _SIGMA * _G * m * T*T*T*T)

def getPressure(T,rho):
	"""
	Calculates the pressure at present location.
	"""
	P = rho * _K * T / (_MU * _M_U) + _A / 3. * T*T*T*T
	return P

def getRho(T,P):
	"""
	Calculates the density at present location.
	Returns density in units of [g cm^-3].
	"""
	# Radiation constant
#	a = 4 * _SIGMA/ _C

	rho = _MU * _M_U / (_K * T) * (P - _A/3. * T*T*T*T)
	return rho

def drdm(r,rho):
	"""
	Calculates the right-hand side of dr/dm.
	"""
	return 1./(4 * np.pi * r * r * rho)

def dPdm(r,m):
	"""
	Calculates the right-hand side of dP/dm.
	"""
	return - _G * m / (4 * np.pi * r*r*r*r)

def dLdm(T,rho):
	"""
	Calculates the right-hand side of dL/dm.
	"""
	epsilon = energyGeneration(T,rho)[0]
	return epsilon
#	return energyGeneration(T,rho) Doesn't work when returning several objects

def dTdm(T,L,r,kappa):
	"""
	Calculates the right-hand side of dT/dm.
	"""
	return -3 * kappa * L / (256 * np.pi*np.pi * _SIGMA * r*r*r*r * T*T*T)


def integration():
	"""
	Function that integrates the equations governing
	the structure of the Sun, including radiative and
	convective energy transfer.
	"""
	
	L0 = 3.84e33				# Units of [erg s^-1]		CGS
	R0 = 6.96e10				# Units of [cm]				CGS
	M0 = 1.99e33				# Units of [g]				CGS
	T0 = 5770
	rho0 = 4.0e-7				# Units of [g cm^-3]		CGS
	P0 = getPressure(T0,rho0)	# Units of [Ba]				CGS

	rho = rho0
	m = M0
	r = R0
	P = P0
	L = L0
	T = T0
	kappa = opacity(T,rho)
	epsilon, PP1, PP2, PP3 = energyGeneration(T, rho)

	dm = -1e20

	# Assuming we only have radiative flux in the beginning, until we have convection.
	F = total_flux(L,r)
	F_R = total_flux(L,r)
	F_C = 0.0

	outfile = open('data.txt','w')
	outfile.write('%3.2e %3.2e %4.0f %3.2e %3.2e %3.2e\n' % \
			(M0, R0, T0, L0, P0, rho0))
	outfile.write('%7.2e %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f ' % \
		(dm, m/M0, r/R0, T/T0, L/L0, P/P0, rho/rho0, epsilon))
	outfile.write('%12.4f %12.4f ' % \
			(F_R/F, F_C/F))
	outfile.write('%12.4f %12.4f %12.4f\n' % \
			(PP1/epsilon, PP2/epsilon, PP3/epsilon))

	start = time.time()
	i = 0
	while m > 0:
		# Progress counter
		percent = 100 - (m / float(M0)) * 100
		sys.stdout.write('Progress: %4.2f %s\r' % (percent, '%'))
		sys.stdout.flush()

		dr = drdm(r,rho) * dm
		dP = dPdm(r,m) * dm
		dL = dLdm(T,rho) * dm
		dT = dTdm(T,L,r,kappa) * dm

		T_new = T + dT	# Superflous?

		F = total_flux(L,r)
		F_R = total_flux(L,r)
		F_C = 0.0

		# Specific heat capacity
		c_p = (5./2) * _K / (_MU * _M_U)	# Units of [erg K^-1 g^-1]

		delta = 1.0		# 1 for ideal gas (maybe revisit later for a general expression)

		# Temperature gradient in adiabatic process
		nabla_ad = P * delta / (T * rho * c_p)
#		print "nabla_ad: ",nabla_ad

		# Temperature gradient when only considering radiation
		nabla_rad = nabla_radiation(T,P,L,m,kappa)
#		print "nabla_rad: ", nabla_rad

		# Checks the instability criterion and calculates dT based on convection if true.
#		if False:
		if nabla_rad > nabla_ad:

			# Parameter between 0.5 and 2. Choose 1 for simplicity
			alpha = 1.0

			# Current acceleration of gravity
			g = _G * m / (r*r)

			# Pressure scale height
			H_p = P / (g * rho)
#			print "H_p: ",H_p

			# Internal energy
			U = (64 * _SIGMA * T*T*T) / ( 3 * kappa * rho*rho * c_p) * np.sqrt(H_p / (g *
				delta))
			print U
#			print "U: ",U
			
			# Mixing length
			l = alpha * H_p
#			print "l: ",l
			
			# Parameters for solving 2nd and 3rd order polynomials
			R = U / (l*l)
#			print "R: ",R
			K = 4 * R
			nabla_diff = nabla_rad - nabla_ad
#			print "nabla_rad - nabla_ad: ", (nabla_rad - nabla_ad)
#			R2 = R*R
			
#			X = ((np.sqrt((27./R2)**2*nabla_diff**2+1836./R2*nabla_diff+6480)\
#					+34+27./R2*nabla_diff)/2.)**(1./3)

#			xi = R/3*(X-11./X-1)

			coeff = [1, R, R*K, -R*nabla_diff]
			xi_roots = np.roots(coeff)
			for root in xi_roots:
				if np.imag(root) == min(abs(np.imag(xi_roots))):
					xi = np.real(root)
					break
#			print "xi: ",xi
			
			nabla = xi*xi+ K*xi+ nabla_ad

			dT = T/P * nabla * dP

			F_R = radiative_flux(m,r,T,P,kappa,nabla)
			F_C = convective_flux(F,F_R)
				
		r_new = r + dr
		P_new = P + dP
		L_new = L + dL
		T_new = T + dT
		rho_new = getRho(T_new,P_new)
		epsilon_new, PP1_new, PP2_new, PP3_new= energyGeneration(T_new, rho_new)
		kappa_new = opacity(T_new, rho_new)
		m += dm

		if r_new < 0 or L_new < 0 or rho_new < 0 or P_new < 0 or T_new < 0:
			print "Something went below zero. Stopped."
			print "Step:", i
			print "Mass:", m/M0
			print "Radius:", r_new/R0
			print "Luminosity:", L_new/L0
			print "Density:", rho_new/rho0
			print "Pressure:", P_new/P0
			print "Temperature:", T_new/T0
			break

#		if dr/r < 0.2 and dT/T < 0.1 and dL/L < 0.1 and dP/P < 0.1:
#			dm_new = - min([abs(L/dLdm(T,rho)), abs(r/drdm(r,rho)),
#			abs(T/dTdm(T,L,r,opacity(T,rho))), abs(P/dPdm(r,m))])
#			if dm_new < 1.1*dm:
#				dm = dm_new
#			else:


		dm_max = - min([0.2*abs(L/dLdm(T,rho)), 0.3*abs(r/drdm(r,rho)),
				0.2*abs(T/(dT/dm)),
				0.2*abs(P/dPdm(r,m))])
		dm_min = - min([0.1*abs(L/dLdm(T,rho)), 0.2*abs(r/drdm(r,rho)),
			0.1*abs(T/(dT/dm)), 0.1*abs(P/dPdm(r,m))])

		while abs(dm) > abs(dm_max) or abs(dm) < abs(dm_min):
			if abs(dm) > abs(dm_max):
				dm *= 0.95
			if abs(dm) < abs(dm_min):
				dm *= 1.05

		r = r_new
		P = P_new
		L = L_new
		T = T_new
		rho = rho_new
		kappa = kappa_new
		epsilon = epsilon_new
		PP1 = PP1_new
		PP2 = PP2_new
		PP3 = PP3_new

		outfile.write('%7.2e %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f ' % \
				(dm, m/M0, r/R0, T/T0, L/L0, P/P0, rho/rho0, epsilon))
		outfile.write('%12.4f %12.4f ' % \
				(F_R/F, F_C/F))
		outfile.write('%12.4f %12.4f %12.4f\n' % \
				(PP1/epsilon, PP2/epsilon, PP3/epsilon))

		print "dm: ",dm 

			
	# Writing elapsed time upon completion
	finish = time.time()
	print "Time elapsed:", ((finish-start)/60.), " min"

	print "r: ",r/R0
	print "m: ",m/M0
	print "P: ",P
	print "L: ",L
	print "T: ",T
	print "rho: ",rho
	print "epsilon: ",epsilon


	return None #r, P, L, T, m, rho, rho0, P0, T0, L0, M0, R0, epsilon, end
integration()


