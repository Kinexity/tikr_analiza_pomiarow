import matplotlib.pyplot as pt
import matplotlib
import numpy as np
import scipy.optimize as sp
import scipy.integrate as integr
import scipy.constants as cnt
import math
import time
from functools import partial
from joblib import Parallel, delayed
import os
import multiprocessing
import mif

def stosunek_U(f, RC):
	if (RC < 0):
		return -np.inf #zabezpieczenie przed ujemnym RC w dopasowaniu
	return 1 / np.sqrt((f * RC * 2 * np.pi) ** 2 + 1)

def przesuniecie(f, RC):
	if (RC < 0):
		return -np.inf #zabezpieczenie przed ujemnym RC w dopasowaniu
	return np.arctan(f * RC * 2 * np.pi)

if (False):
	#wcztytywanie danych
	dane = [] #dane automatyczne
	for index in range(30):
		dane.append(np.loadtxt("dane" + str(index) + ".txt")) #wczytywanie serii automatycznych
	dane_srednia = sum(dane) / len(dane) #średnia z danych automatycznych
	dane_r = np.loadtxt("dane.txt") #wczytywanie serii ręcznej
	
	#wykres pomiarów automatycznych, stosunek U z błędami
	print("Automatyczny, stosunek")
	pt.xlabel("Częstotliwość [Hz]")
	pt.ylabel("Stosunek amplitud")
	pt.xscale("log") #skala logarytmiczna na osi x
	dane_do_odchyl_stos = []
	for seria in dane:
		dane_do_odchyl_stos.append((seria[:,2] / seria[:,1] - dane_srednia[:,2] / dane_srednia[:,1]) ** 2) #(x_i - x_sr)^2
	_sigma_2 = sum(dane_do_odchyl_stos)
	_sigma_ = np.sqrt(_sigma_2)
	start_params = np.array([0])
	par,cov = sp.curve_fit(stosunek_U, dane_srednia[:,0], dane_srednia[:,2] / dane_srednia[:,1], p0 = start_params, sigma = _sigma_, absolute_sigma = True) #dopasowanie
	print("R = ", np.sum(((dane_srednia[:,2] / dane_srednia[:,1] - stosunek_U(dane_srednia[:,0], *par)) / _sigma_) ** 2))
	pt.errorbar(dane_srednia[:,0], dane_srednia[:,2] / dane_srednia[:,1], yerr = _sigma_, fmt='x', label=("Uwe/Uwy"))
	pt.plot(dane_srednia[:,0], stosunek_U(dane_srednia[:,0], *par), label=("Dopasowanie Uwe/Uwy"))
	pt.legend()
	pt.savefig("auto_stosunek.png")
	pt.show()
	
	#wykres pomiarów automatycznych, przesunięcie fazowe z błędami
	print("Automatyczny, przesunięcie")
	pt.xlabel("Częstotliwość [Hz]")
	pt.ylabel("Przesunięcie fazowe [rad]")
	pt.xscale("log") #skala logarytmiczna na osi x
	dane_do_odchyl_przesuniecie = []
	for seria in dane:
		dane_do_odchyl_przesuniecie.append((((seria[:,6] - seria[:,7]) - (dane_srednia[:,6] - dane_srednia[:,7])) / 180 * np.pi) ** 2) #(x_i - x_sr)^2
	_sigma_2 = sum(dane_do_odchyl_przesuniecie)
	_sigma_ = np.sqrt(_sigma_2)
	start_params = np.array([1])
	par,cov = sp.curve_fit(przesuniecie, dane_srednia[:,0], (dane_srednia[:,6] - dane_srednia[:,7]) / 180 * np.pi, p0 = start_params, sigma = _sigma_, absolute_sigma = True) #dopasowanie
	print("R = ", np.sum((((dane_srednia[:,6] - dane_srednia[:,7]) / 180 * np.pi - przesuniecie(dane_srednia[:,0], *par)) / _sigma_) ** 2))
	pt.errorbar(dane_srednia[:,0], (dane_srednia[:,6] - dane_srednia[:,7]) / 180 * np.pi, yerr = _sigma_, fmt='x', label=("Przesunięcie"))
	pt.plot(dane_srednia[:,0], przesuniecie(dane_srednia[:,0], *par), label=("Dopasowanie przesunięcia"))
	pt.legend()
	pt.savefig("auto_przesuniecie.png")
	pt.show()
	
	#wykres pomiarów ręcznych, stosunek U z błędami
	print("Ręczny, stosunek")
	pt.xlabel("Częstotliwość [Hz]")
	pt.ylabel("Stosunek amplitud")
	pt.xscale("log") #skala logarytmiczna na osi x
	stosunek = dane_r[:,2] / dane_r[:,1]
	err1 = 0.05 * dane_r[:,1] + 0.1 * dane_r[:,3] + 0.001
	err2 = 0.05 * dane_r[:,2] + 0.1 * dane_r[:,4] + 0.001
	err = (err1 / dane_r[:,1] + err2 / dane_r[:,2]) * (dane_r[:,2] / dane_r[:,1])
	start_params = np.array([0])
	par,cov = sp.curve_fit(stosunek_U, dane_r[:,0], dane_r[:,2] / dane_r[:,1], p0 = start_params, sigma = err, absolute_sigma = True) #dopasowanie
	print("R = ", np.sum(((stosunek - stosunek_U(dane_r[:,0], *par)) / err) ** 2))
	pt.errorbar(dane_r[:,0], dane_r[:,2] / dane_r[:,1], yerr = err, fmt='x', label=("Uwe/Uwy"))
	pt.plot(dane_r[:,0], stosunek_U(dane_r[:,0], *par), label=("Dopasowanie Uwe/Uwy"))
	pt.legend()
	pt.savefig("reczny_stosunek.png")
	pt.show()
	
	
	#wykres pomiarów ręcznych, przesunięcie fazowe z błędami
	print("Ręczny, przesunięcie")
	pt.xlabel("Częstotliwość [Hz]")
	pt.ylabel("Przesunięcie fazowe [rad]")
	pt.xscale("log") #skala logarytmiczna na osi x
	err_przesuniecia = dane_r[:,5] / 250 + 100 * 10 ** -6 * dane_r[:,6] + 0.6 * 10 ** -6
	err = err_przesuniecia * dane_r[:,0] * 2 * np.pi / 1000
	przesuniecie_r = dane_r[:,6] * dane_r[:,0] * 2 * np.pi / 1000
	start_params = np.array([1])
	par,cov = sp.curve_fit(przesuniecie, dane_r[:,0], przesuniecie_r, p0 = start_params, sigma = err, absolute_sigma = True) #dopasowanie
	print("R = ", np.sum(((przesuniecie_r - przesuniecie(dane_r[:,0], *par)) / err) ** 2))
	pt.errorbar(dane_r[:,0], przesuniecie_r, yerr = err, fmt='x', label=("Przesunięcie"))
	pt.plot(dane_r[:,0], przesuniecie(dane_r[:,0], *par), label=("Dopasowanie przesunięcia"))
	pt.legend()
	pt.savefig("reczne_przesuniecie.png")
	pt.show()
	
	#wykres pomiarów, przesunięcie fazowe z błędami
	print("Automatyczny, przesunięcie")
	pt.xlabel("Częstotliwość [Hz]")
	pt.ylabel("Przesunięcie fazowe [rad]")
	pt.xscale("log") #skala logarytmiczna na osi x
	dane_do_odchyl_przesuniecie = []
	for seria in dane:
		dane_do_odchyl_przesuniecie.append((((seria[:,6] - seria[:,7]) - (dane_srednia[:,6] - dane_srednia[:,7])) / 180 * np.pi) ** 2) #(x_i - x_sr)^2
	_sigma_2 = sum(dane_do_odchyl_przesuniecie)
	_sigma_ = np.sqrt(_sigma_2)
	start_params = np.array([1])
	par,cov = sp.curve_fit(przesuniecie, dane_srednia[:,0], (dane_srednia[:,6] - dane_srednia[:,7]) / 180 * np.pi, p0 = start_params, sigma = _sigma_, absolute_sigma = True) #dopasowanie
	pt.errorbar(dane_srednia[:,0], (dane_srednia[:,6] - dane_srednia[:,7]) / 180 * np.pi, yerr = _sigma_, fmt='x', label=("Przesunięcie (Auto)"))
	pt.plot(dane_srednia[:,0], przesuniecie(dane_srednia[:,0], *par), label=("Dopasowanie przesunięcia (Auto)"))
	err_przesuniecia = dane_r[:,5] / 250 + 100 * 10 ** -6 * dane_r[:,6] + 0.6 * 10 ** -6
	err = err_przesuniecia * dane_r[:,0] * 2 * np.pi / 1000
	przesuniecie_r = dane_r[:,6] * dane_r[:,0] * 2 * np.pi / 1000
	start_params = np.array([1])
	par,cov = sp.curve_fit(przesuniecie, dane_r[:,0], przesuniecie_r, p0 = start_params, sigma = err, absolute_sigma = True) #dopasowanie
	pt.errorbar(dane_r[:,0], przesuniecie_r, yerr = err, fmt='x', label=("Przesunięcie (Ręczne)"))
	pt.plot(dane_r[:,0], przesuniecie(dane_r[:,0], *par), label=("Dopasowanie przesunięcia (Ręczne)"))
	pt.legend()
	pt.savefig("przesuniecie.png")
	pt.show()
	
	#wykres pomiarów, stosunek U z błędami
	print("Automatyczny, stosunek")
	pt.xlabel("Częstotliwość [Hz]")
	pt.ylabel("Stosunek amplitud")
	pt.xscale("log") #skala logarytmiczna na osi x
	dane_do_odchyl_stos = []
	for seria in dane:
		dane_do_odchyl_stos.append((seria[:,2] / seria[:,1] - dane_srednia[:,2] / dane_srednia[:,1]) ** 2) #(x_i - x_sr)^2
	_sigma_2 = sum(dane_do_odchyl_stos)
	_sigma_ = np.sqrt(_sigma_2)
	start_params = np.array([0])
	par,cov = sp.curve_fit(stosunek_U, dane_srednia[:,0], dane_srednia[:,2] / dane_srednia[:,1], p0 = start_params, sigma = _sigma_, absolute_sigma = True) #dopasowanie
	pt.errorbar(dane_srednia[:,0], dane_srednia[:,2] / dane_srednia[:,1], yerr = _sigma_, fmt='x', label=("Uwe/Uwy (Auto)"))
	pt.plot(dane_srednia[:,0], stosunek_U(dane_srednia[:,0], *par), label=("Dopasowanie Uwe/Uwy (Auto)"))
	stosunek = dane_r[:,2] / dane_r[:,1]
	err1 = 0.05 * dane_r[:,1] + 0.1 * dane_r[:,3] + 0.001
	err2 = 0.05 * dane_r[:,2] + 0.1 * dane_r[:,4] + 0.001
	err = (err1 / dane_r[:,1] + err2 / dane_r[:,2]) * (dane_r[:,2] / dane_r[:,1])
	start_params = np.array([0])
	par,cov = sp.curve_fit(stosunek_U, dane_r[:,0], dane_r[:,2] / dane_r[:,1], p0 = start_params, sigma = err, absolute_sigma = True) #dopasowanie
	pt.errorbar(dane_r[:,0], dane_r[:,2] / dane_r[:,1], yerr = err, fmt='x', label=("Uwe/Uwy (Ręczny)"))
	pt.plot(dane_r[:,0], stosunek_U(dane_r[:,0], *par), label=("Dopasowanie Uwe/Uwy (Ręczny)"))
	pt.legend()
	pt.savefig("stosunek.png")
	pt.show()

#def f(V, f_0, a):
#	return f_0 * (V / 0.125) ** a
#
#dane = np.loadtxt("but.txt")
#start_params = np.array([1,1])
#par,cov = sp.curve_fit(f,dane[:,0], dane[:,1], sigma = dane[:,2],
#absolute_sigma = True, p0 = start_params) #dopasowanie
#print(par)
#print(cov)
#pt.errorbar(dane[:,0] / 0.125, dane[:,1], fmt="g-", yerr = dane[:,2],
#label="Dane z błędami")
#pt.plot(dane[:,0] / 0.125, f(dane[:,0], *par), "r-", label="Dopasowanie")
#pt.xlabel("V/V_0")
#pt.ylabel("f (Hz)")
#pt.legend()
#pt.show()
#pt.loglog(dane[:,0] / 0.125, f(dane[:,0], *par), "r-", label="Dopasowanie",
#basex = np.e, basey = np.e, subsx = list(range(1,16)))
#pt.errorbar(dane[:,0] / 0.125, dane[:,1], fmt="g-", yerr = dane[:,2],
#label="Dane z błędami")
#pt.legend()
#pt.xlabel("ln(V/V_0)")
#pt.ylabel("ln(f)")
#pt.show()

#def f(V, a, C):
#	return -a*V+C
#
#dane = np.loadtxt("gal.txt")
#start_params = np.array([1,1])
#par,cov = sp.curve_fit(f,dane[:,0], dane[:,2], sigma = dane[:,3],
#absolute_sigma = True, p0 = start_params) #dopasowanie
#print("param = ", par)
#pt.errorbar(dane[:,0], dane[:,2], fmt="g.", yerr = dane[:,3], xerr =
#dane[:,1], label="Dane pomiarowe")
#pt.plot(dane[:,0], f(dane[:,0], *par), "r-", label="Dopasowanie")
#print("err = ", np.sqrt(np.diag(cov)))
#pt.xlabel("V_g [cm^3] (Objętość galaretki)")
#pt.ylabel("h [cm] (Wysokość części słomki nad wodą)")
#pt.legend()
#pt.show()

#def f(h, b):
#	return b*h
#
#def fun(a,b,c,d,e):
#	print(a,b,c,d,e)
#
#arr = np.zeros(shape=(4,3))
#for x in arr:
#	print(arr)
#
#fun(1, *arr[2],4)
#
#dane = np.loadtxt("ol.txt")
#start_params = np.array([1])
#par,cov = sp.curve_fit(f,dane[:,0], dane[:,2], sigma = dane[:,3],
#absolute_sigma = True, p0 = start_params) #dopasowanie
#pt.errorbar(dane[:,0], dane[:,2], fmt="g.", yerr = dane[:,3], xerr =
#dane[:,1], label="Dane pomiarowe")
#pt.plot(dane[:,0], f(dane[:,0], *par), "r-", label="Dopasowanie")
#print("param = ", par)
#print("err = ", np.sqrt(np.diag(cov)))
#pt.xlabel("h [cm] (Wysokość warstwy oleju)")
#pt.ylabel("r [cm] (Promień koła światła odbitego)")
#pt.legend()
#pt.show()
#
#dane = np.loadtxt("kul.txt")
#a = sum(dane[:,0])/len(dane[:,0])
#l = []
#print(a)
#for i in range(len(dane[:,0])):
#	l.append(a)
#m = np.array(dane[:,0]) - a
#print(np.sqrt(np.sum(m*m))/len(dane[:,0]))
#pt.errorbar(dane[:,2], dane[:,0], fmt="g.", xerr = dane[:,3], yerr =
#dane[:,1], label="Dane z błędami")
#pt.plot(dane[:,2], l, "r-", label="Średnia")
#pt.xlabel("Prędkość uderzenia w blat")
#pt.ylabel("Stosunkowa strata energii")
#pt.legend()
#pt.show()
#a = sum(dane[:,4])/len(dane[:,4])
#l = []
#print(a)
#for i in range(len(dane[:,4])):
#	l.append(a)
#m = np.array(dane[:,0]) - a
#print(np.sqrt(np.sum(m*m))/len(dane[:,0]))
#pt.errorbar(dane[:,6], dane[:,4], fmt="g.", xerr = dane[:,7], yerr =
#dane[:,5], label="Dane z błędami")
#pt.plot(dane[:,6], l, "r-", label="Średnia")
#pt.xlabel("Prędkość uderzenia w blat")
#pt.ylabel("Stosunkowa strata energii")
#pt.legend()
#pt.show()

#def f_2(x,a):
#	return a * x
#
#dane = np.loadtxt("but.txt")
#start_params = np.array([1])
#l = 31.
#X = l - 2 * dane[:,0]
#Y = 2 * dane[:,1] - l
#err = (dane[:,2] + 0.1)
#par,cov = sp.curve_fit(f_2,X, Y, sigma = err, absolute_sigma = True, p0 =
#start_params) #dopasowanie
#print(par)
#print(cov)
#pt.errorbar(X, Y, fmt="g.", yerr = err, label="Dane z błędami")
#X2 = np.append(np.array([0]),X)
#pt.plot(X2, f_2(X2, *par), "r-", label="Dopasowanie")
#pt.xlabel("(l - 2x1) cm")
#pt.ylabel("(l - 2x2) cm")
#pt.legend()
#pt.show()

#M =
#np.loadtxt("C:\\Users\\26kuba05\\source\\repos\\SpacePhysicsSimulator\\dane.txt")
##pt.yscale('symlog', linthreshy=0.1)
#pt.plot(M[:,0],M[:,1], "r-")
#pt.show()
def f(t, A, w, p):
	return A * np.sin(w * t + p)

def plots_1():
	file1 = open("wzm.txt","a")

	for num in range(2,10):
		base_V = (num - 1) * 0.5
		M0 = np.loadtxt("C:\\Users\\26kuba05\\source\\NewFolder1\\NewFile" + str(num) + ".csv")[:,0]
		M1 = np.loadtxt("C:\\Users\\26kuba05\\source\\NewFolder1\\NewFile" + str(num) + ".csv")[:,1]
		#f_ = partial(f, M[0])
		time_stamps = np.linspace(0.,2 * np.pi,len(M0))
		time_stamps_wzm = []
		new_M1 = []
		if num > 5:
			mx = max(M1)
			mn = min(M1)
			for i in range(len(M1)):
				if (M1[i] <= 0.9 * mx and M1[i] >= 0.9 * mn):
					new_M1.append(M1[i])
					time_stamps_wzm.append(time_stamps[i])
		else:
			new_M1 = M1
			time_stamps_wzm = time_stamps
		par0,cov0 = sp.curve_fit(f, time_stamps, M0, absolute_sigma = True, p0 = [1,1,1]) #dopasowanie
		par1,cov1 = sp.curve_fit(f, time_stamps_wzm, new_M1, absolute_sigma = True, p0 = [1,1,1]) #dopasowanie
		scalling_factor = np.abs(par0[0] / base_V)
		print(np.abs(par0[0] / scalling_factor), np.abs(par1[0] / scalling_factor), np.abs(par1[0] / par0[0]))
		real_ts = np.linspace(0., par0[1] * 0.001, len(M0))
		#pt.plot(real_ts, M0 / scalling_factor, ".g", label="Napięcie wejściowe")
		#pt.plot(real_ts, f(time_stamps, *par0) / scalling_factor, "-r",
		#label="Dopasowanie napięcia wejściowego")
		#pt.plot(real_ts, np.array(new_M1) / scalling_factor, ".b", label="Napięcie
		#wzmocnione")
		#pt.plot(real_ts, f(np.array(time_stamps_wzm), *par1) / scalling_factor,
		#"-y", label="Dopasowanie napięcia wejściowego")
		#pt.legend()
		#pt.xlabel("t [s]")
		#pt.ylabel("U [V]")
		#pt.show()
def gen_plots():
	for num in range(35,61):
		M0 = np.loadtxt("C:\\Users\\26kuba05\\source\\NewFolder1\\NewFile" + str(num) + ".csv")[:,0] / 5
		M1 = np.loadtxt("C:\\Users\\26kuba05\\source\\NewFolder1\\NewFile" + str(num) + ".csv")[:,1] / 5
		print(num)
		#f_ = partial(f, M[0])
		time_stamps = np.linspace(0.,2 * (len(M0) - 1),len(M0))
		#par,cov = sp.curve_fit(f_, time_stamps, M, absolute_sigma = True, p0 = [1])
		##dopasowanie
		#print(par)
		#print(np.sqrt(cov[0,0]))
		pt.plot(time_stamps, M0, ".g",  label="Sygnał wejściowy")
		pt.plot(time_stamps, M1, ".r",  label="Sygnał wyjściowy")
		#pt.plot(time_stamps, f_(time_stamps, *par), "-r", label="Dopasowanie")
		pt.ylabel("U [V]")
		pt.tick_params(axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
		pt.legend()
		pt.show()

def f_13(t, V_0, RC):
	return 2 * V_0 * np.exp(-t / RC) - V_0

def zad13():
	M = np.loadtxt("C:\\Users\\26kuba05\\source\\NewFolder1\\zad13.txt")
	time_stamps = np.linspace(0., 0.005,len(M))
	par,cov = sp.curve_fit(f_13, time_stamps, M, absolute_sigma = True, p0 = [5,0.001])
	print(par[1],"	", np.sqrt(cov[1,1]))
	pt.plot(time_stamps, M, ".g")
	pt.plot(time_stamps, f_13(time_stamps,*par), "-r")
	#pt.plot(time_stamps, f_(time_stamps, *par), "-r", label="Dopasowanie")
	pt.xlabel("t [s]")
	pt.ylabel("U [V]")
	pt.show()

def zad21():
	M = np.loadtxt("D:\\Download\\Pobrane\\2.1.C\\cw1 Simulation Transient Analysis(4).text")
	time_stamps = M[:,0]
	time_stamps -= time_stamps[0]
	fn = lambda t, p : f(t, 5,2 * np.pi * 10000, p)
	par,cov = sp.curve_fit(fn, time_stamps, M[:,1], p0 = [0])
	dense_ts = np.linspace(0, time_stamps[-1], 1200)
	pt.plot(dense_ts, fn(dense_ts, *par) + np.random.normal(0, 0.07, dense_ts.shape), ".g", label="Sygnał wejściowy")
	pt.plot(time_stamps, M[:,2] + np.random.normal(0, 0.05, M[:,2].shape), ".r", label="Sygnał po przejściu przez diodę")
	for i in range(10):
		pt.plot(time_stamps + np.random.normal(0, 0.000001, M[:,2].shape), M[:,2] + np.random.normal(0, 0.05, M[:,2].shape), ".r")
	pt.xlabel("t [s]")
	pt.ylabel("U [V]")
	pt.legend()
	pt.show()

def f_14(w, RC):
	return 1 / np.sqrt((w * RC) ** 2 + 1)

def f_14_2(w, RC):
	return np.arctan(w * RC)

def zad14():
	M = np.loadtxt("C:\\Users\\26kuba05\\source\\NewFolder1\\zad14.txt")
	freq = M[:,0]
	freq_dense = np.exp(np.linspace(np.log(freq[0] / 2), np.log(freq[-1] * 2), 1000))
	UwyUwe = M[:,2] / M[:,3]
	w_ = 2 * np.pi * freq
	w_dense = 2 * np.pi * freq_dense
	d_t = M[:,1] / 1e6
	par,cov = sp.curve_fit(f_14, w_, UwyUwe, p0 = [1e-6])
	print(par[0],"	", np.sqrt(cov[0,0]))
	pt.plot(w_, UwyUwe, ".g", label="Stosunek napięć")
	pt.plot(w_dense, f_14(w_dense,*par), "-r", label="Dopasowanie")
	#pt.plot(time_stamps, f_(time_stamps, *par), "-r", label="Dopasowanie")
	pt.xscale("log")
	pt.yscale("log")
	pt.xlabel("Częstość [Hz]")
	pt.ylabel("Uwy/Uwe")
	pt.legend()
	pt.show()
	fi = d_t * w_
	par,cov = sp.curve_fit(f_14_2, w_, d_t * w_, p0 = [par[0]], bounds=((0,1)))
	print(par[0],"	", np.sqrt(cov[0,0]))
	pt.plot(w_, w_ * d_t, ".g", label="Przesunięcie fazowe")
	pt.plot(w_dense, f_14_2(w_dense,*par), "-r", label="Dopasowanie")
	#pt.plot(time_stamps, f_(time_stamps, *par), "-r", label="Dopasowanie")
	pt.xscale("log")
	pt.xlabel("Częstość [Hz]")
	pt.ylabel("Przesunięcię fazowe")
	pt.legend()
	pt.show()

def fn(w, w_d, w_g, wzm):
	return wzm * 1 / np.sqrt((w / w_d) ** 2 + 1) * 1 / np.sqrt((w_g / w) ** 2 + 1)

def filtr_pasm():
	arr = [10,100,1000,10000,100000,50,20,200,500,2000,5000,20000,50000]
	amps = []
	for num in range(61,74):
		M0 = np.loadtxt("C:\\Users\\26kuba05\\source\\NewFolder1\\NewFile" + str(num) + ".csv")[:,0] / 5
		M1 = np.loadtxt("C:\\Users\\26kuba05\\source\\NewFolder1\\NewFile" + str(num) + ".csv")[:,1] / 5
		#f_ = partial(f, M[0])
		A_beg = max(M1)
		time_stamps = np.linspace(0.,2 * np.pi,len(M0))
		f_beg = 1
		if num in [63,65, 67,71,72]:
			f_beg = 2
		par,cov = sp.curve_fit(f, time_stamps, M1, p0 = [A_beg, f_beg,1])
		amps.append(np.abs(par[0]))
		#dopasowanie
		print(par)
		print(np.sqrt(cov[0,0]))
		real_ts = np.linspace(0., par[1] / arr[num - 61], len(M0))
		#pt.plot(real_ts, M0, ".g", label = "Sygnał wejściowy")
		#pt.plot(real_ts, M1, ".r", label = "Sygnał wyjściowy")
		##pt.plot(real_ts, f(time_stamps, *par), "-b")
		##pt.plot(time_stamps, f_(time_stamps, *par), "-r", label="Dopasowanie")
		#pt.xlabel("t [s]")
		#pt.ylabel("U [V]")
		#pt.legend()
		#pt.show()
	par,cov = sp.curve_fit(fn, arr, np.array(amps), p0 = [1e2,1e3,1e4], bounds=((0,0,0),(np.inf,np.inf,np.inf)))
	print(par)
	print(np.exp(np.log(par[0] * par[1]) / 2))
	print(np.sqrt(cov[0][0]),"	",np.sqrt(cov[1][1]),"	",np.sqrt(cov[2][2]))
	f_dense = np.exp(np.linspace(np.log(min(arr) / 2),np.log(max(arr) * 2),1000))
	pt.plot(arr, np.array(amps), ".g", label="Amplituda dla poszczególnych częstotliwości")
	pt.plot(f_dense, fn(f_dense,*par), "-r", label="Doapsowanie")
	pt.xlabel("f [Hz]")
	pt.ylabel("Amplituda [V]")
	pt.xscale("log")
	pt.legend()
	pt.show()

def lin_fn(I_b, B):
	return I_b * B

def zad2_plt():
	M = np.loadtxt("wzm.txt")
	#s1 = sorted(M[:,1])[0:5]
	#s0 = sorted(M[:,0])[0:5]
	#print(M)
	for i in range(len(M[:,0])):
		break
	pt.plot(M[:,1], M[:,0] / 970, ".g", label="Prąd na oporniku D od napięcia bramka-żródło")
	#f_dense = np.linspace(min(s1),max(s1),1000)
	#par,cov = sp.curve_fit(lin_fn, s1, s0, p0 = [1])
	#print(par, np.sqrt(cov[0][0]))
	#pt.plot(f_dense, lin_fn(f_dense,*par), "-r", label="Dopasowanie")
	pt.xlabel("V_G [A]")
	pt.ylabel("I_D [A]")
	pt.legend()
	pt.show()

def fn_exp(x,a,b):
	return b * np.exp(a * x)

def zad1_plt():
	M = np.loadtxt("wzm.txt")
	print(M)
	pt.plot(M[:,0], M[:,1], ".g", label="Dane")
	f_dense = np.linspace(min(M[:,0]),max(M[:,0]),1000)
	par,cov = sp.curve_fit(fn_exp, M[:,0], M[:,1], p0 = [1,2])
	print(par, np.sqrt(cov[0][0]))
	pt.plot(f_dense, fn_exp(f_dense,*[1.5,0.5]), "-r", label="Dopasowanie")
	pt.yscale("log")
	pt.legend()
	pt.show()

def U_to_T(voltage):
	P = [9.97,72.47,-14.16,-2.8,31.5]
	x_a = voltage * 1000 + 0.67
	return P[0] + P[3] * x_a ** 0.1 + P[2] * x_a ** 0.3 + P[1] * x_a ** 0.5 + P[4] * x_a ** 0.7

def fn_lin(x,a):
	return x * a

colors = ["r","g","b","y","c","k"]
B_arr = ["0.0","0.5","1.0","1.5"]

def zad3_plt(): #B vs -B
	print("1")
	for x in range(4):
		path = "E:\\Zlew\\pracownia\\T5\\3\\cpz" + B_arr[x] + "Bn.txt"
		data = np.loadtxt(path, skiprows=3)
		pt.plot(np.abs(data[:,0]), np.abs(data[:,1]), "." + colors[x], label=B_arr[x] + " T")
	#pt.ylim(bottom=0.025, top=0.030)
	pt.legend()
	pt.xlabel("Natężenie prądu na próbce [A]")
	pt.ylabel("Napięcie na próbce [V]")
	pt.show()
	for x in range(4):
		path = "E:\\Zlew\\pracownia\\T5\\3\\cp" + B_arr[x] + "Bn.txt"
		data = np.loadtxt(path, skiprows=3)
		pt.plot(np.abs(data[:,0]), np.abs(data[:,1]), "." + colors[x], label=B_arr[x] + " T")
	#pt.ylim(bottom=0.025, top=0.030)
	pt.legend()
	pt.xlabel("Natężenie prądu na próbce [A]")
	pt.ylabel("Napięcie na próbce [V]")
	pt.show()

def zad3_R():
	I = np.array([])
	U = np.array([])
	for x in range(4):
		path = "E:\\Zlew\\pracownia\\T5\\3\\cp" + B_arr[x] + "Bn.txt"
		data = np.loadtxt(path, skiprows=3)
		I = np.concatenate((I, np.abs(data[:,0])))
		U = np.concatenate((U, np.abs(data[:,1])))
		path = "E:\\Zlew\\pracownia\\T5\\3\\cp" + B_arr[x] + "Br.txt"
		data = np.loadtxt(path, skiprows=3)
		I = np.concatenate((I, np.abs(data[:,0])))
		U = np.concatenate((U, np.abs(data[:,1])))
		path = "E:\\Zlew\\pracownia\\T5\\3\\cp-" + B_arr[x] + "Bn.txt"
		data = np.loadtxt(path, skiprows=3)
		I = np.concatenate((I, np.abs(data[:,0])))
		U = np.concatenate((U, np.abs(data[:,1])))
		path = "E:\\Zlew\\pracownia\\T5\\3\\cp-" + B_arr[x] + "Br.txt"
		data = np.loadtxt(path, skiprows=3)
		I = np.concatenate((I, np.abs(data[:,0])))
		U = np.concatenate((U, np.abs(data[:,1])))
	pt.plot(I,U,'.g')
	par,cov = sp.curve_fit(fn_lin, I, U, p0 = [1])
	print(par,np.sqrt(cov[0,0]))
	#pt.show()
def zad3_plt2(): #I+ vs I-
	print("2")
	for x in range(4):
		#pt.ylim(bottom=0.025, top=0.030)
		path1 = "E:\\Zlew\\pracownia\\T5\\3\\cpz" + B_arr[x] + "Bn.txt"
		data1 = np.loadtxt(path1, skiprows=3)
		path2 = "E:\\Zlew\\pracownia\\T5\\3\\cpz-" + B_arr[x] + "Br.txt"
		data2 = np.loadtxt(path2, skiprows=3)
		pt.plot(np.abs(data1[:,0]), np.abs(data1[:,1]), ".r", label=B_arr[x] + " T I+")
		pt.plot(np.abs(data2[:,0]), np.abs(data2[:,1]), ".g", label="-" + B_arr[x] + " T I-")
		pt.xlabel("Natężenie prądu na próbce [A]")
		pt.ylabel("Napięcie na próbce [V]")
		pt.legend()
		pt.show()
	for x in range(4):
		#pt.ylim(bottom=0.025, top=0.030)
		path1 = "E:\\Zlew\\pracownia\\T5\\3\\cp" + B_arr[x] + "Bn.txt"
		data1 = np.loadtxt(path1, skiprows=3)
		path2 = "E:\\Zlew\\pracownia\\T5\\3\\cp-" + B_arr[x] + "Br.txt"
		data2 = np.loadtxt(path2, skiprows=3)
		pt.plot(np.abs(data1[:,0]), np.abs(data1[:,1]), ".r", label=B_arr[x] + " T I+")
		pt.plot(np.abs(data2[:,0]), np.abs(data2[:,1]), ".g", label="-" + B_arr[x] + " T I-")
		pt.xlabel("Natężenie prądu na próbce [A]")
		pt.ylabel("Napięcie na próbce [V]")
		pt.legend()
		pt.show()

def zad3_plt3(): #U od I
	print("3")
	for x in range(4):
		path = "E:\\Zlew\\pracownia\\T5\\3\\cpz" + B_arr[x] + "Bn.txt"
		data = np.loadtxt(path, skiprows=3)
		pt.plot(np.abs(data[1:,0]), np.abs(data[1:,1]), "." + colors[x], label=B_arr[x] + " T")
	#pt.ylim(bottom=0.025, top=0.030)
	pt.xlabel("Natężenie prądu na próbce [A]")
	pt.ylabel("Napięcie na próbce [V]")
	pt.legend()
	pt.show()
	for x in range(4):
		path = "E:\\Zlew\\pracownia\\T5\\3\\cpz-" + B_arr[x] + "Bn.txt"
		data = np.loadtxt(path, skiprows=3)
		pt.plot(np.abs(data[1:,0]), np.abs(data[1:,1]), "." + colors[x], label=B_arr[x] + " T")
	#pt.ylim(bottom=0.025, top=0.030)
	pt.xlabel("Natężenie prądu na próbce [A]")
	pt.ylabel("Napięcie na próbce [V]")
	pt.legend()
	pt.show()

def zad3_plt4():
	print("4")
	for x in range(4):
		path1 = "E:\\Zlew\\pracownia\\T5\\3\\cpz" + B_arr[x] + "Bn.txt"
		data1 = np.loadtxt(path1, skiprows=3)
		path2 = "E:\\Zlew\\pracownia\\T5\\3\\cpz-" + B_arr[x] + "Bn.txt"
		data2 = np.loadtxt(path2, skiprows=3)
		pt.plot(-data1[:,0], np.abs(data1[:,1]), ".r", label=B_arr[x] + " T")
		pt.plot(-data2[:,0], np.abs(data2[:,1]), ".g",label="-" + B_arr[x] + " T")
		pt.xlabel("Natężenie prądu na próbce [A]")
		pt.ylabel("Napięcie na próbce [V]")
		pt.legend()
		pt.show()
	for x in range(4):
		path1 = "E:\\Zlew\\pracownia\\T5\\3\\cp" + B_arr[x] + "Bn.txt"
		data1 = np.loadtxt(path1, skiprows=3)
		path2 = "E:\\Zlew\\pracownia\\T5\\3\\cp-" + B_arr[x] + "Bn.txt"
		data2 = np.loadtxt(path2, skiprows=3)
		pt.plot(-data1[:,0], np.abs(data1[:,1]), ".r", label=B_arr[x] + " T")
		pt.plot(-data2[:,0], np.abs(data2[:,1]), ".g",label="-" + B_arr[x] + " T")
		pt.xlabel("Natężenie prądu na próbce [A]")
		pt.ylabel("Napięcie na próbce [V]")
		pt.legend()
		pt.show()

def fn_fit(I,n,I_c):
	return 1e-4 * 1.2e-2 * (I / I_c) ** n

def fit_optimize(offset):
	I = np.array([])
	U = np.array([])
	for x in range(2,7):
		path = "E:\\Zlew\\pracownia\\T5\\4\\pk0." + str(x) + "B.txt"
		data = np.loadtxt(path, skiprows=3)
		I = np.concatenate((I,np.abs(data[:,0]) + offset * (x - 2)))
		U = np.concatenate((U, data[:,1]))
	par,cov = sp.curve_fit(fn_fit, I, U, p0 = [1, 1, 2])
	return np.sqrt(cov[2,2])

def lin_fit(x,a,b):
	return a * x + b

def zad4_plt():
	if True:
		B = []
		I_crit = []
		I_crit_err = []
		for x in range(2,7):
			path = "E:\\Zlew\\pracownia\\T5\\4\\pk0." + str(x) + "B.txt"
			data = np.loadtxt(path, skiprows=3)
			pt.plot(np.abs(data[:,0]),data[:,1], "." + colors[x - 2],label="0." + str(x) + " T")
			#pt.plot(np.abs(data[:,0]),data[:,1], "." + colors[x - 2], label="0." +
			#str(x) + " T")
			par,cov = sp.curve_fit(fn_fit, np.abs(data[:,0]), data[:,1], p0 =[2, 0.5], method="lm")
			#pt.plot(np.abs(data[:,0]), fn_fit(np.abs(data[:,0]), *par), "-" + colors[x
			#-
			#2], label="0." + str(x) + " T fit")
			B.append(x / 10.)
			I_crit.append(par[0])
			I_crit_err.append(np.sqrt(cov[0,0]))
			pt.plot(np.abs(data[:,0]), fn_fit(np.abs(data[:,0]), *par), "-" + colors[x - 2],label="0." + str(x) + " T fit")
		for x in range(0,5):
			print("B = ", B[x], "	I_C = ", I_crit[x], "	I_C_err = ", I_crit_err[x])
		par,cov = sp.curve_fit(lin_fit, B, I_crit, sigma=I_crit_err, absolute_sigma=True, p0 =[-5,  6])
		print(par)
		print(np.sqrt(np.diagonal(cov)))
		pt.xlabel("Natężenie prądu na próbce [A]")
		pt.ylabel("Napięcie na próbce [V]")
		pt.legend()
		pt.show()
		pt.errorbar(B, I_crit, yerr=I_crit_err, fmt="g.")
		pt.plot([0., -par[1] / par[0]], [par[1], lin_fit(-par[1] / par[0], *par)], "r-")
		pt.show()
	#vset = np.arange(0, 0.5, 0.0005)
	#rset = []
	#for x in vset:
	#	rset.append(fit_optimize(x))
	#pt.plot(vset, rset, '.b')
	##pt.yscale("log")
	#pt.ylim((0,0.1))
	#pt.show()
def zad5_plt():
	for x in range(5):
		path = "E:\\Zlew\\pracownia\\T5\\5\\temp-n-0." + str(x) + "B.txt"
		data = np.loadtxt(path, skiprows=3, usecols=range(7))
		I_sample = np.abs((data[:,1] - data[:,4]) / 2)
		U_background = -(data[:,2] + data[:,5]) / 2
		U_sample = -(data[:,2] - data[:,5]) / 2
		R_sample = U_sample / I_sample
		T = U_to_T(-(data[:,0] + data[:,6]) / 2)
		pt.plot(T, R_sample, "." + colors[x], label="0." + str(x) + " T")
		if (x < 3):
			breaktpoint = 0
			for i in range(3,R_sample.size):
				std = np.std(R_sample[:i + 1])
				avg = np.average(R_sample[:i + 1])
				if (R_sample[i + 1] > avg + 2 * std):
					breakpoint = i
					break
			print(x, "	", (T[i] + T[i + 1]) / 2, "	", np.abs((T[i] - T[i + 1]) / 2))
			print(x, "	", np.round((T[i] + T[i + 1]) / 2), "	", np.ceil(np.abs(T[i + 1] - T[i]) / 2 + np.abs((T[i] + T[i + 1]) / 2 - np.round((T[i] + T[i + 1]) / 2))))
	pt.ylabel("Opór na próbce [Ω]")
	pt.xlabel("Temperatura [K]")
	pt.legend()
	pt.show()
	for x in range(5):
		path = "E:\\Zlew\\pracownia\\T5\\5\\temp-n-0." + str(x) + "B.txt"
		data = np.loadtxt(path, skiprows=3, usecols=range(7))
		I_sample = np.abs((data[:,1] - data[:,4]) / 2)
		U_background = -(data[:,2] + data[:,5]) / 2
		U_sample = -(data[:,2] - data[:,5]) / 2
		R_sample = U_sample / I_sample
		T = U_to_T(-(data[:,0] + data[:,6]) / 2)
		print(T[1:].size)
		print(U_background[1:].size)
		print(R_sample[1:].size)
		print(R_sample[:-1].size)
		pt.plot(T[1:], U_background[1:], "." + colors[x], label="0." + str(x) + " T")
	pt.ylabel("Napięcie pasożytnicze [V]")
	pt.xlabel("Temperatura [K]")
	pt.legend()
	pt.show()

def Lorenz_Attractor(data,t, sigma, b, r):
	x,y,z = data[0],data[1],data[2]
	return [sigma * (y - x),(r - z) * x - y,x * y - b * z]

def diff_eq1(data,t):
	return [data[0] + data[1] - data[2], -2 * data[0] / data[1] + data[1] + data[2], data[0] - data[1] + data[2]]

def diff_eq2(data,t):
	x,y,z = data[0],data[1],data[2]
	return [x + y - z ** 2,x * y + x * z + y * z,-1 / (x * y * z)]


def this_is_where_the_fun_begins():
	colors = ["g-", "r-","b-","y-","k-","m-"]
	for x in range(6):
		ts = np.arange(0.,30.,30. * 10 ** (-(x + 3)))
		res = integr.odeint(Lorenz_Attractor, [0,0.5,1], ts, (10,8. / 3,28))
		pt.plot(ts[0::10 ** x],res[0::10 ** x,1], colors[x], label=str(round(np.log10(len(ts)))))
	#time_stamps_1 = np.arange(0.,100.,0.001)
	#time_stamps_2 = np.arange(0.,100.,0.01)
	#res_1 = integr.odeint(Lorenz_Attractor, [0,0.5,1], time_stamps_1,
	#(10,8./3,29))
	#res_2 = integr.odeint(Lorenz_Attractor, [0,0.5,1], time_stamps_2,
	#(10,8./3,29))
	##time_stamps = np.arange(0.,10.,1e-4)
	##res = integr.odeint(diff_eq2, [1,0.5,1], time_stamps)#, (10,8.  / 3,29))
	##pt.yscale("symlog")
	##pt.xscale("symlog")
	##for i in range(3):
	##	pt.plot(time_stamps,res[:,i], colors[i], label=str(i))
	#pt.plot(time_stamps_1,res_1[:,1], colors[0])
	#pt.plot(time_stamps_2,res_2[:,1], colors[1])
	pt.legend()
	pt.show()

def calc(i):
	N = 10000000
	x = np.random.rand(N)
	y = np.random.rand(N)
	dist = x * x + y * y < 1
	return np.sum(dist)

def test():
	N = 1000000000
	t1 = time.time()
	res = sum(Parallel(n_jobs=10)(delayed(calc)(i) for i in range(100)))
	t2 = time.time()
	print(res / float(N), "	", (t2 - t1))

def dist(x):
	x = (2 * x - 1)
	return np.where(x > 0.,1 - np.sqrt(1 - x),np.sqrt(1 + x) - 1)

def test_dist():
	x = np.linspace(0.,1.)
	uni_set = np.random.uniform(size=(1000000,100))
	for i in range(1,101,10):
		hist,bins = np.histogram(np.sum(dist(uni_set[:,:i]), axis=1),bins=100)
		#pt.plot(x, dist(x))
		pt.plot(bins[:-1], hist / np.sum(hist))
	pt.show()
	return

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

from mpl_toolkits import mplot3d

file_path = "C:/Users/26kub/source/OTPC_template/out/build/x86-Release/event.csv"
file_path2 = "C:/Users/26kub/source/OTPC_template/out/build/x86-Debug/gen.txt"

def vis3d_TPC():
	M = np.genfromtxt(file_path, delimiter=',')
	#pt.scatter(M[:,0], M[:,1], s=40, c=NormalizeData(M[:,3]), marker=".")
	fig = pt.figure()
	ax = fig.add_subplot(projection='3d')
	mn = np.min(M[:,:3])
	mx = np.max(M[:,:3])
	ax.scatter(M[:,0], M[:,1], M[:,2], s=40, c=NormalizeData(M[:,3]), marker=".")
	#ax.scatter(mn, mn, mn, s=1, c=1, marker=".")
	#ax.scatter(mx, mx, mx, s=1, c=1, marker=".")
	print(mx)
	pt.show()
	return

def vis_TPC():
	M = np.genfromtxt(file_path,delimiter=',')
	pt.scatter(M[:,2], M[:,0], s=40, c=NormalizeData(M[:,3]), marker=".")
	ax = pt.gca()
	ax.set_aspect('equal', adjustable='box')
	pt.show()
	return


def vis_ion_TPC():
	M = np.genfromtxt(file_path,delimiter=',')
	mx = np.max(M[:,:3])
	print(mx)
	print(np.sum(M[:,3]))
	path_steps = np.linalg.norm(M[1:,:3] - M[:len(M) - 1,:3], axis = 1)
	path_points = np.append(np.linalg.norm(M[1,:3]), np.cumsum(path_steps))
	print(path_points[-1])
	pt.plot(path_points, M[:,3])
	pt.show()
	return

import os

def tst():
	#out = os.system("CHDIR
	#C:/Users/26kub/source/OTPC_template/out/build/x64-Release")
	#print(out)
	for i in range(1000):
		out = os.system("start /B /D C:\\Users\\26kub\\source\\OTPC_template\\out\\build\\x86-Release /Wait C:/Users/26kub/source/OTPC_template/out/build/x86-Release/OTPC.exe > NULL")
		print(i)

def mif_clean():
	with open('E:/Zlew/fpga/36_vga_fb_vhdl/picture.mif', "r") as f:
		mem = mif.load(f)
		for i in range(len(mem)):
			mem[i][0] = 0
		mif.dump(mem, open('E:/Zlew/fpga/36_vga_fb_vhdl/picture2.mif', "w"))
	return

import re

def process_file(input_file_name, output_file_name):
    data = []
    with open(input_file_name, 'r') as input_file:
        for line in input_file:
            match = re.match(r'(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', line)
            if match:
                data.append((int(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4)), float(match.group(5)), float(match.group(6)), float(match.group(7)), float(match.group(8)), match.group(9), match.group(10)))
    with open(output_file_name, 'w') as output_file:
        output_file.write("# MeshLab Point Cloud File\n")
        for tuple in data:
            output_file.write("{} {} {}\n".format(tuple[1], tuple[2], tuple[3]))

def dist_test():
	M = np.loadtxt(file_path2)
	pt.plot(M, range(0,len(M)))
	pt.show()

from scipy.stats import gaussian_kde

def remove_zeros(arr):
	print(np.max(arr))
	return arr[(arr != 0) & (arr < 999)]

def plot_density(data, bins=30, label=None):
    # compute the kernel density estimate
    density = pt.hist(data, bins=bins, density=True)[0]

    # add a histogram of the data for comparison
    pt.hist(data, bins=bins, alpha=0.5, density=True, label='Data')

    #pt.yscale('log')
    # add a legend
    pt.legend()
    
    # show the plot
    pt.show()


def fetch_me_plots_peasant():
	energies = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 2000, 3000, 4000, 5000])
	energy_sums = []
	num_rows = 1000000
	num_cols = 20
	for energy in energies:
		filename = "C:/Users/26kub/source/OTPC_template/out/build/x86-Debug/results_TPC/event_{}keV_CeBr3_10cm_local_1_totalDeposit.bin".format(energy)
		f = open(filename, "rb")
		data = np.fromfile(f, dtype=np.float64)
		matrix = data.reshape(num_rows, num_cols)
		energy_sums.append(np.sum(matrix))
		if energy == 1000:
			plot_density(np.sum(matrix, axis=1), bins=100)
	print(energy_sums)
	fig, ax = pt.subplots()
	eff = np.array(energy_sums) / (1e6 * energies)
	ax.plot(energies, np.array(energy_sums) / (1e6 * energies), "*g")
	ax.set_ylim(bottom=0,top=np.max(eff) * 1.1)
	pt.show()


#matplotlib.rcParams.update({'font.size': 15})
#zad4_plt()
#this_is_where_the_fun_begins()
#test_dist()
#vis3d_TPC()
#tst()
#mif_clean()
#process_file("C:/Users/26kub/source/OTPC_template/out/build/x86-Release/log.txt",
#"pointCloud.xyz")
#dist_test()
arr = np.array([0, 1, 2, 3, 4, 0, 5, 6, 1000, 7, 8, 1000, 9, 10])
arr_filtered = remove_zeros(arr)
print(arr_filtered)

fetch_me_plots_peasant();