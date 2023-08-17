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
import glob
import re

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

def dist_test():
	M = np.loadtxt(file_path2)
	pt.plot(M, range(0,len(M)))
	pt.show()

def remove_zeros(arr):
	return arr[(arr != 0)]

def remove_nonpositive_elements(arr):
    return arr[(arr > 0)]

def generate_dense_points(x_sparse, y_sparse, num_points):
    x_dense = np.linspace(min(x_sparse), max(x_sparse), num_points)
    y_dense = np.interp(x_dense, x_sparse, y_sparse)
    return x_dense, y_dense

from scipy.interpolate import CubicSpline

def generate_dense_points2(x_sparse, y_sparse, num_points):
    spline = CubicSpline(x_sparse, y_sparse)
    x_dense = np.linspace(min(x_sparse), max(x_sparse), num_points)
    y_dense = spline(x_dense)
    return x_dense, y_dense

def plot_density(data, bins=30, label=None, yl=None, xl=None):
	# compute the kernel density estimate
	density = pt.hist(data, bins=bins, density=True, alpha = 0.)[0]
	
	# add a histogram of the data for comparison
	pt.hist(data, bins=bins, histtype='step', color="black", alpha=0.5, density=True, label=label)
	
	pt.xlabel(xl)
	pt.ylabel(yl)
	
	pt.yscale('log')
	# add a legend
	pt.legend()
	
	# show the plot
	pt.show()


def plot_density2(data1, data2, bins=30, label1=None, label2=None):
    # compute the kernel density estimate for data1
    density1 = pt.hist(data1, bins=bins, density=True,histtype='step',  alpha=0.5, label=label1)[0]

    # compute the kernel density estimate for data2
    density2 = pt.hist(data2, bins=bins, density=True,histtype='step',  alpha=0.5, label=label2)[0]

    # set the y-axis scale to log
    pt.yscale('log')

    # add a legend
    pt.legend()

    # show the plot
    pt.show()

def plot_density3(data1, data2, data3, bins_=30, label1=None, label2=None, label3=None):
    # compute the histogram for all data sets to obtain common bins
    _, bins = np.histogram(np.concatenate((data1, data2, data3)), bins=bins_)

    # compute the kernel density estimate for data1
    density1 = pt.hist(data1, bins=bins, density=False, histtype='step', label=label1)[0]

    # compute the kernel density estimate for data2
    density2 = pt.hist(data2, bins=bins, density=False, histtype='step', label=label2)[0]

    # compute the kernel density estimate for data3
    density3 = pt.hist(data3, bins=bins, density=False, histtype='step', label=label3)[0]

    # set the y-axis scale to log
    pt.yscale('log')

    # add a legend
    pt.legend()

    # show the plot
    pt.show()

import matplotlib as mpl

def plot_density_2d(data1, data2, bins=30, label1=None, label2=None):
    # Create a two-dimensional histogram
    pt.hist2d(data1, data2, bins=bins, cmap='rainbow', norm=mpl.colors.LogNorm())

    # Set the y-axis scale to log
    #pt.yscale('log')

    # Add a colorbar
    pt.colorbar()

    # Set labels and legend
    pt.xlabel(label1)
    pt.ylabel(label2)
    pt.legend([label1, label2])

    # Show the plot
    #pt.show()
def plot_density2_same_bins(data1, data2, bins=30, label1=None, label2=None, leg_loc='upper left'):
    # compute the histogram bins
    hist_range = (min(min(data1), min(data2)), max(max(data1), max(data2)))
    #hist_bins = pt.hist(data1, bins=bins, range=hist_range)[1]

    # compute the kernel density estimate for data1
    density1 = pt.hist(data1, bins=bins,histtype='step',  density=False, alpha=0.5, label=label1, range=hist_range)[0]

    # compute the kernel density estimate for data2
    density2 = pt.hist(data2, bins=bins,histtype='step', density=False, alpha=0.5, label=label2, range=hist_range)[0]

    # set the y-axis scale to log
    pt.yscale('log')

    # add a legend
    pt.legend(loc=leg_loc)



num_rows = 10000
num_cols = 20
energies = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 2000, 3000, 4000, 5000])
groups = {0:[10,11,16,17],1:[8,9,14,15],2:[12,13,18,19],3:[0,1,4,5],4:[2,3,6,7]}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ffa07a', '#6a5acd']
scintillator_list = ["CeBr3", "LaBr3Ce"]
detector_lenght_list = [5, 7.5,10]
cut_value_list = [1,0.1,0.01]
symbols = ['o', 's', '^', 'v', 'D', 'p', '*', 'x', '+', 'h']
physics_list_list = ["local", "G4EmLivermore", "G4EmPenelope"]

def applyEnergyResolutionCeBr3(data):
    # Constants for FWHM calculation
    a = 8.135
    b = 2.893e-2
    c = -5.062e-6

    # Calculate FWHM for each energy point
    fwhm = a + b * data + c * data ** 2

    # Apply random shift to each data point
    shifted_data = np.where(data > 0, np.random.normal(data, fwhm / 2.355), 0)

    # Zero out negative shifted points
    shifted_data = np.where(shifted_data < 0, 0, shifted_data)

    return shifted_data

def applyEnergyResolutionLaBr3(data):
    # Constants for FWHM calculation
    a = 6.206
    b = 2.754e-2
    c = -4.071e-6

    # Calculate FWHM for each energy point
    fwhm = a + b * data + c * data ** 2

    # Apply random shift to each data point
    shifted_data = np.where(data > 0, np.random.normal(data, fwhm / 2.355), 0)

    # Zero out negative shifted points
    shifted_data = np.where(shifted_data < 0, 0, shifted_data)

    return shifted_data

def applyEnergyResolution(data, material_name):
	if material_name == "CeBr3":
		return applyEnergyResolutionCeBr3(data)
	elif material_name == "LaBr3Ce":
		return applyEnergyResolutionLaBr3(data)

def applyEnergyResolutionGasOTPC(data):

    # Calculate FWHM for each energy point
    fwhm = 200 #a + b * data + c * data**2

    # Apply random shift to each data point
    shifted_data = np.where(data > 0, np.random.normal(data, fwhm / 2.355), 0)

    # Zero out negative shifted points
    shifted_data = np.where(shifted_data < 0, 0, shifted_data)

    return shifted_data

#load simulation data from file based on parameters
def load_data(scintilator_type, crystal_depth, phys_list, cut, data_index, energy, optional_string=""):
	filename = "C:/Users/26kub/results_TPC/event_{}_{}cm_{}_{}mm{}_{}/".format(scintilator_type, crystal_depth, phys_list, cut, optional_string, data_index) + "event_{}keV_{}_{}cm_{}_{}mm{}_totalDeposit.bin".format(energy, scintilator_type, crystal_depth, phys_list, cut, optional_string)
	f = open(filename, "rb")
	data = np.fromfile(f, dtype=np.float64)
	data = data[:len(data) // num_cols * num_cols]
	return data.reshape(len(data) // num_cols, num_cols)

#load simulation data from file based on parameters
def load_gas_data(scintilator_type, crystal_depth, phys_list, cut, data_index, energy, optional_string=""):
	filename = "C:/Users/26kub/results_TPC/event_{}_{}cm_{}_{}mm{}_{}/".format(scintilator_type, crystal_depth, phys_list, cut, optional_string, data_index) + "event_{}keV_{}_{}cm_{}_{}mm{}_totalDeposit_gas.bin".format(energy, scintilator_type, crystal_depth, phys_list, cut, optional_string)
	f = open(filename, "rb")
	data = np.fromfile(f, dtype=np.float64)
	return data.reshape(len(data) // 1, 1)

def load_both(scintilator_type, crystal_depth, phys_list, cut, data_index, energy, optional_string=""):
	filename = "C:/Users/26kub/results_TPC/event_{}_{}cm_{}_{}mm{}_{}/".format(scintilator_type, crystal_depth, phys_list, cut, optional_string, data_index) + "event_{}keV_{}_{}cm_{}_{}mm{}_totalDeposit.bin".format(energy, scintilator_type, crystal_depth, phys_list, cut, optional_string)
	f = open(filename, "rb")
	data = np.fromfile(f, dtype=np.float64)
	filename2 = "C:/Users/26kub/results_TPC/event_{}_{}cm_{}_{}mm{}_{}/".format(scintilator_type, crystal_depth, phys_list, cut, optional_string, data_index) + "event_{}keV_{}_{}cm_{}_{}mm{}_totalDeposit_gas.bin".format(energy, scintilator_type, crystal_depth, phys_list, cut, optional_string)
	f2 = open(filename2, "rb")
	data2 = np.fromfile(f2, dtype=np.float64)
	print(len(data) // num_cols,len(data2))
	len_actual = np.min([len(data) // num_cols,len(data2)])
	len_actual -= len_actual % num_rows
	data2 = data2[:len_actual]
	data = data[:len_actual * num_cols]
	return data.reshape(len(data) // num_cols, num_cols), data2.reshape(len(data2) // 1, 1)

#extract subset of data from crystals from only one group
def extract_group_data(data, group_index):
	return data[:,groups[group_index]]

def extract_crystal_data(data, crystal_index):
	return data[:,crystal_index]

# only count events where energy was deposited on one crystal
def count_full_events_per_crystal(data, energy):
	return np.sum(data >= energy * 0.999)

# county events where energy deposited on the whole group (group_data is data
# from one group!)
def count_full_events_per_group(group_data, energy):
	data = np.sum(group_data, axis = 1)
	return np.sum(data >= energy * 0.999)


def count_full_events_w_gas(data, gas_data, energy, gas_energy):
	return np.sum(np.logical_and(data >= energy * 0.999, gas_data >= gas_energy * 0.999))

def fetch_me_plots_peasant():
	vals = scintillator_list
	srcs = len(vals)
	simulation_indexes = [0,0,0]
	energy_sums = [[] for i in range(srcs)]
	full_events = [[[] for j in range(20)] for i in range(srcs)]
	mtx = [np.array([1]) for i in range(srcs)]
	for energy in energies:
		for i in range(srcs):
			data = load_data(vals[i],10, "G4EmLivermore", 0.01, simulation_indexes[i], energy)
			energy_sums[i].append(np.sum(data))
			for j in range(20):
				#matrix = extract_group_data(data, j)
				matrix = data[:, j]
				#full_events[i][j].append(count_full_events_per_group(matrix, energy))
				full_events[i][j].append(count_full_events_per_crystal(matrix, energy))
			if energy == 5000:
				mtx[i] = remove_zeros(np.sum(data, axis=1))
	#plot_density2_same_bins(mtx[0],mtx[1],100,"CeBr3","LaBr3")
	#print(full_events)
	#print(full_events2)
	#pt.title("no isotopes vs isotopes")
	#plot_density2(mtx1,mtx2,100,"no isotopes","isotopes")
	fig, ax = pt.subplots()
	#eff = np.array(energy_sums) / (1e6 * energies)
	#ax.plot(energies, eff, colors[0], label="CeBr3")
	#eff = np.array(energy_sums2) / (1e6 * energies)
	#ax.plot(energies, eff, colors[1], label="LaBr3")
	#for i in range(srcs):
	#	for j in range(5):
	#		ax.plot(energies, np.array(full_events[i][j]) / 1e6, colors[j] +
	#		symbols[i], label="group {} {}".format(j, vals[i]))
	for i in range(srcs):
		#y_vals = np.array(energy_sums[i]) / (1e6 * energies)
		y_vals = np.sum(np.array(full_events[i]), axis = 0) / 1e6
		ax.scatter(energies, y_vals, color = colors[i], marker = symbols[i], label="{}".format(vals[i]))
		ax.plot(*generate_dense_points2(energies, y_vals, 1000), color = colors[i], linestyle = "-")
	#eff2 = np.array(energy_sums2) / (1e6 * energies)
	#ax.plot(energies,eff2, ".r", label="isotopes")
	#ax.plot(energies, np.array(full_events2) / 1e6, ".y", label="local cnt")
	ax.set_ylim(bottom=0)#,top=np.max(np.array(full_events[1:]) / 1e6) * 1.1)
	pt.title("total effs")
	pt.ylabel("Efficiency")
	pt.xlabel("Photon energy [keV]")
	pt.legend()
	pt.show()

def remove_zero_fields(arr1, arr2):
    non_zero_indices = np.nonzero(arr1)
    return arr1[non_zero_indices], arr2[non_zero_indices]

def remove_zero_fields_either(arr1, arr2):
	print(len(arr1), len(arr2))
	non_zero_indices1 = (arr1 != 0)
	non_zero_indices2 = (arr2 != 0)
	non_zero_indices = np.logical_or(non_zero_indices1, non_zero_indices2)
	return arr1[non_zero_indices], arr2[non_zero_indices]

def remove_nonpositive_fields_both(arr1, arr2):
	pos1 = (arr1 > 0)
	pos2 = (arr2 > 0)
	pos = np.logical_and(pos1,pos2) #np.intersect1d(pos1, pos2)
	return arr1[pos], arr2[pos]

def fetch_me_plots_peasant5():
	energy = 583
	data = np.array([])
	gas_data = np.array([])
	for i in range(1,6):#(31, 47):
		dt, gdt = load_both("CeBr3",10, "G4EmLivermore", 0.01, i, energy, "_dataFile")
		data = np.concatenate((data, np.sum(dt, axis=1)), axis = 0)
		gas_data = np.concatenate((gas_data, np.sum(gdt,axis=1)), axis = 0)
	#dt, gdt = load_both("CeBr3",10, "G4EmLivermore", 0.01, 0, energy,
	#"_dataFile")
	#data = np.sum(dt, axis=1)
	#gas_data = np.sum(gdt, axis=1)
	print("Final events: ", len(gas_data), len(data))
	#cnt = count_full_events_w_gas(np.sum(data, axis=1), np.sum(gas_data, axis=1),
	#583, 595)
	#print(cnt)
	#print(cnt/1e6)
	pt.title("Energy deposit [keV]")
	data2, gas_data2 = remove_zero_fields(data, gas_data)
	data3, gas_data3 = remove_nonpositive_fields_both(data, gas_data)
	data4, gas_data4 = remove_nonpositive_fields_both(applyEnergyResolutionCeBr3(data3), applyEnergyResolutionGasOTPC(gas_data3))
	gas_data_full_res_filtered = remove_nonpositive_elements(applyEnergyResolutionGasOTPC(remove_nonpositive_elements(gas_data)))
	#plot_density_2d(data4, gas_data4,100,"crystals","gas")
	#plot_density(gas_data, 100, "gas deposit")
	plot_density2_same_bins(gas_data4, gas_data_full_res_filtered, 100, "all gas events", "gas events with crystal deposition")
	plot_density3(remove_zeros(data), remove_nonpositive_elements(applyEnergyResolutionCeBr3(remove_zeros(data))), remove_nonpositive_elements(applyEnergyResolutionLaBr3(remove_zeros(data))), 100, "crystal depo, no res", "crystal depo, with CeBr3 res", "crystal depo, with LaBr3 res")

dpi = 200
ticks_rotation = 90
inch_sizes = [12,6]
use_titles = False

def cut_value_comparison():
	full_events = {cut_value:[] for cut_value in cut_value_list}
	for energy in energies:
		for cut_value in cut_value_list:
			data = load_data("CeBr3",10, "G4EmLivermore", cut_value, 0, energy)
			total_counts = np.sum([count_full_events_per_crystal(extract_crystal_data(data, i), energy) for i in range(20)])
			full_events[cut_value].append(total_counts)
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	ax.set_ylim(bottom=0,top=np.max(np.array(max(max(lst) for lst in full_events.values())) / 1e6) * 1.1)
	for cut_value in cut_value_list:
		i = cut_value_list.index(cut_value)
		y_vals = np.array(full_events[cut_value]) / 1e6
		ax.scatter(energies, y_vals, color = colors[i], marker=symbols[i], label=str(cut_value) + " mm")
		ax.plot(*generate_dense_points2(energies, y_vals, 1000), color = colors[i], linestyle = "-")
	for energy in energies:
		ax.axvline(energy, color='gray', linestyle='-', alpha=0.3)
	pt.xticks(energies, rotation = ticks_rotation)
	if use_titles:
		pt.title("Całkowita wydajność w zależności od parametru cut value")
	pt.ylabel("Wydajność")
	pt.xlabel("Energia fotonów [keV]")
	pt.legend(loc='upper right')
	pt.savefig("cut_value_comparison.png", dpi = dpi, bbox_inches='tight')
	return

def physics_list_comparison():
	vals_list = physics_list_list
	full_events = {val:[] for val in vals_list}
	for energy in energies:
		for val in vals_list:
			data = load_data("CeBr3",10, val, 0.01, 0, energy)
			total_counts = np.sum([count_full_events_per_crystal(extract_crystal_data(data, i), energy) for i in range(20)])
			full_events[val].append(total_counts)
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	ax.set_ylim(bottom=0,top=np.max(np.array(max(max(lst) for lst in full_events.values())) / 1e6) * 1.1)
	for val in vals_list:
		i = vals_list.index(val)
		y_vals = np.array(full_events[val]) / 1e6
		ax.scatter(energies, y_vals, color = colors[i], marker=symbols[i], label=str(val))
		ax.plot(*generate_dense_points2(energies, y_vals, 1000), color = colors[i], linestyle = "-")
	for energy in energies:
		ax.axvline(energy, color='gray', linestyle='-', alpha=0.3)
	pt.xticks(energies, rotation = ticks_rotation)
	if use_titles:
		pt.title("Całkowita wydajność w zależności od parametru physics list")
	pt.ylabel("Wydajność")
	pt.xlabel("Energia fotonów [keV]")
	pt.legend(loc='upper right')
	pt.savefig("physics_list_comparison.png", dpi = dpi, bbox_inches='tight')
	return

def scintillator_comparison():
	vals_list = scintillator_list
	full_events = {val:[] for val in vals_list}
	for energy in energies:
		for val in vals_list:
			data = load_data(val,10, "G4EmLivermore", 0.01, 0, energy)
			total_counts = np.sum([count_full_events_per_crystal(extract_crystal_data(data, i), energy) for i in range(20)])
			full_events[val].append(total_counts)
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	ax.set_ylim(bottom=0,top=np.max(np.array(max(max(lst) for lst in full_events.values())) / 1e6) * 1.1)
	for val in vals_list:
		i = vals_list.index(val)
		y_vals = np.array(full_events[val]) / 1e6
		ax.scatter(energies, y_vals, color = colors[i], marker=symbols[i], label=str(val))
		ax.plot(*generate_dense_points2(energies, y_vals, 1000), color = colors[i], linestyle = "-")
	for energy in energies:
		ax.axvline(energy, color='gray', linestyle='-', alpha=0.3)
	pt.xticks(energies, rotation = ticks_rotation)
	if use_titles:
		pt.title("Całkowita wydajność w zależności od materiału scyntylacyjnego")
	pt.ylabel("Wydajność")
	pt.xlabel("Energia fotonów [keV]")
	pt.legend(loc='upper right')
	pt.savefig("scintillator_comparison.png", dpi = dpi, bbox_inches='tight')
	return

def efficiency_of_photon_catching_comparison():
	vals_list = [48, 50]
	full_events = {val:[] for val in vals_list}	
	lbls = {48:"Ułamek fotonów z detektorem",50:"Ułamek fotonów bez detektora OTPC"}
	for energy in energies:
		for val in vals_list:
			data = load_data("CeBr3",10, "G4EmLivermore", 0.01, val, energy)
			total_counts = np.sum(np.logical_or.reduce(data > 0, axis = 1))
			full_events[val].append(total_counts)
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	ax.set_ylim(bottom=0,top=np.max(np.array(max(max(lst) for lst in full_events.values())) / 1e6) * 1.1)
	for val in vals_list:
		i = vals_list.index(val)
		y_vals = np.array(full_events[val]) / 1e6
		ax.scatter(energies, y_vals, color = colors[i], marker=symbols[i], label=lbls[val])
		ax.plot(*generate_dense_points2(energies, y_vals, 1000), color = colors[i], linestyle = "-")
	for energy in energies:
		ax.axvline(energy, color='gray', linestyle='-', alpha=0.3)
	pt.xticks(energies, rotation = ticks_rotation)
	if use_titles:
		pt.title("Ułamek fotonów, które zdeponowały jakąkolwiek energię w zależności od materiału scyntylacyjnego")
	pt.ylabel("Ułamek fotonów")
	pt.xlabel("Energia fotonów [keV]")
	pt.legend(loc='upper right')
	pt.savefig("efficiency_of_photon_catching_comparison.png", dpi = dpi, bbox_inches='tight')
	return

def detector_shadow_comparison():
	vals_list = [48, 50]
	lbls = {48:"Wydajność z detektorem OTPC",50:"Wydajność bez detektora OTPC"}
	full_events = {val:[] for val in vals_list}
	for energy in energies:
		for val in vals_list:
			data = load_data("CeBr3",10, "G4EmLivermore", 0.01, val, energy)
			total_counts = np.sum([count_full_events_per_crystal(extract_crystal_data(data, i), energy) for i in range(20)])
			full_events[val].append(total_counts)
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	ax.set_ylim(bottom=0,top=np.max(np.array(max(max(lst) for lst in full_events.values())) / 1e6) * 1.1)
	for val in vals_list:
		i = vals_list.index(val)
		y_vals = np.array(full_events[val]) / 1e6
		ax.scatter(energies, y_vals, color = colors[i], marker=symbols[i], label=lbls[val])
		ax.plot(*generate_dense_points2(energies, y_vals, 1000), color = colors[i], linestyle = "-")
	for energy in energies:
		ax.axvline(energy, color='gray', linestyle='-', alpha=0.3)
	pt.xticks(energies, rotation = ticks_rotation)
	if use_titles:
		pt.title("Ułamek fotonów, które zdeponowały jakąkolwiek energię w zależności od materiału scyntylacyjnego")
	pt.ylabel("Wydajność")
	pt.xlabel("Energia fotonów [keV]")
	pt.legend(loc='upper right')
	pt.savefig("detector_shadow_comparison.png", dpi = dpi, bbox_inches='tight')
	return

def detector_lenght_comparison():
	vals_list = detector_lenght_list
	full_events = {val:[] for val in vals_list}
	for energy in energies:
		for val in vals_list:
			data = load_data("CeBr3",val, "G4EmLivermore", 0.01, 0, energy)
			total_counts = np.sum([count_full_events_per_crystal(extract_crystal_data(data, i), energy) for i in range(20)])
			full_events[val].append(total_counts)
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	ax.set_ylim(bottom=0,top=np.max(np.array(max(max(lst) for lst in full_events.values())) / 1e6) * 1.1)
	for val in vals_list:
		i = vals_list.index(val)
		y_vals = np.array(full_events[val]) / 1e6
		ax.scatter(energies, y_vals, color = colors[i], marker=symbols[i], label=str(val) + " cm")
		ax.plot(*generate_dense_points2(energies, y_vals, 1000), color = colors[i], linestyle = "-")
	for energy in energies:
		ax.axvline(energy, color='gray', linestyle='-', alpha=0.3)
	pt.xticks(energies, rotation = ticks_rotation)
	if use_titles:
		pt.title("Całkowita wydajność w zależności od długości detektorów gamma")
	pt.ylabel("Wydajność")
	pt.xlabel("Energia fotonów [keV]")
	pt.legend(loc='upper right')
	pt.savefig("detector_lenght_comparison.png", dpi = dpi, bbox_inches='tight')
	return

def counting_comparison():
	vals_list = [True, False]
	full_events = {val:[] for val in vals_list}
	full_events_all = []
	for energy in energies:
		data = load_data("CeBr3",10, "G4EmLivermore", 0.01, 0, energy)
		total_counts_crystals = np.sum([count_full_events_per_crystal(extract_crystal_data(data, i), energy) for i in range(20)])
		total_counts_groups = np.sum([count_full_events_per_group(extract_group_data(data, i), energy) for i in range(5)])
		full_events[False].append(total_counts_crystals)
		full_events[True].append(total_counts_groups)
		full_events_all.append(count_full_events_per_crystal(np.sum(data, axis = 1), energy))
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	ax.set_ylim(bottom=0,top=np.max(np.array(max(max(lst) for lst in full_events.values())) / 1e6) * 1.1)
	for val in vals_list:
		i = vals_list.index(val)
		y_vals = np.array(full_events[val]) / 1e6
		lbl = ""
		if val:
			lbl = "zliczanie grupowe"
		else:
			lbl = "zliczanie pojedyncze"
		ax.scatter(energies, y_vals, color = colors[i], marker=symbols[i], label=lbl)
		ax.plot(*generate_dense_points2(energies, y_vals, 1000), color = colors[i], linestyle = "-")
	if True:
		i = 2
		y_vals = np.array(full_events_all) / 1e6
		lbl = "zliczanie całkowite"
		ax.scatter(energies, y_vals, color = colors[i], marker=symbols[i], label=lbl)
		ax.plot(*generate_dense_points2(energies, y_vals, 1000), color = colors[i], linestyle = "-")
	for energy in energies:
		ax.axvline(energy, color='gray', linestyle='-', alpha=0.3)
	pt.xticks(energies, rotation = ticks_rotation)
	if use_titles:
		pt.title("Całkowita wydajność w zależności od metody zliczania energii zdeponowanej")
	pt.ylabel("Wydajność")
	pt.xlabel("Energia fotonów [keV]")
	pt.legend(loc='upper right')
	pt.savefig("counting_comparison.png", dpi = dpi, bbox_inches='tight')
	pt.clf()

def groups_comparison():
	vals_list = list(range(5))
	full_events = {val:[] for val in vals_list}
	for energy in energies:
		data = load_data("CeBr3",10, "G4EmLivermore", 0.01, 0, energy)
		for val in vals_list:
			total_counts = np.sum([count_full_events_per_group(extract_group_data(data, val), energy)])
			full_events[val].append(total_counts)
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	ax.set_ylim(bottom=0,top=np.max(np.array(max(max(lst) for lst in full_events.values())) / 1e6) * 1.1)
	for val in vals_list:
		i = vals_list.index(val)
		y_vals = np.array(full_events[val]) / 1e6
		ax.scatter(energies, y_vals, color = colors[i], marker=symbols[i], label="Grupa detektorów " + str(val))
		ax.plot(*generate_dense_points2(energies, y_vals, 1000), color = colors[i], linestyle = "-")
	for energy in energies:
		ax.axvline(energy, color='gray', linestyle='-', alpha=0.3)
	#y_ticks = ax.get_yticks()
	#for y_tick in y_ticks:
		#ax.axhline(y=y_tick, color='gray', alpha=0.5)
	pt.xticks(energies, rotation = ticks_rotation)
	if use_titles:
		pt.title("Całkowita wydajność na grupę")
	pt.ylabel("Wydajność")
	pt.xlabel("Energia fotonów [keV]")
	pt.legend(loc='upper right')
	pt.savefig("group_comparison.png", dpi = dpi, bbox_inches='tight')
	return

def find_keys_with_extreme_maximal_values(data):
    max_value = float('-inf')
    min_value = float('inf')
    max_key = None
    min_key = None

    for key, value in data.items():
        max_val = max(value)

        if max_val > max_value:
            max_key = key
            max_value = max_val

        if max_val < min_value:
            min_key = key
            min_value = max_val

    return max_key, min_key

positions_dict = {"+100":"Najwyższa wydajność","+111":"Najniższa wydajność","+000":"Wydajność ze środka"}

def position_comparison():
	vals_list = []
	digits = ['0', '1']
	signs = ['+0', "+1", '-1']
	for sign in signs:
	    for digit1 in digits:
	        for digit2 in digits:
	            vals_list.append(sign + digit1 + digit2)
	full_events = {val:[] for val in vals_list}
	for energy in energies:
		for val in vals_list:
			data = load_data("CeBr3",10, "G4EmLivermore", 0.01, 0, energy, "_" + val)
			total_counts = np.sum([count_full_events_per_crystal(extract_crystal_data(data, i), energy) for i in range(20)])
			full_events[val].append(total_counts)
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	ax.set_ylim(bottom=0,top=np.max(np.array(max(max(lst) for lst in full_events.values())) / 1e6) * 1.1)
	max_m,min_m = find_keys_with_extreme_maximal_values(full_events)
	new_vals = [max_m, min_m, "+000"]
	for val in new_vals:
		i = new_vals.index(val)
		y_vals = np.array(full_events[val]) / 1e6
		ax.scatter(energies, y_vals, color = colors[i], marker=symbols[i], label=str(positions_dict[val]))
		ax.plot(*generate_dense_points2(energies, y_vals, 1000), color = colors[i], linestyle = "-")
	print(np.min(np.array(full_events[min_m]) / 1e6))
	for energy in energies:
		ax.axvline(energy, color='gray', linestyle='-', alpha=0.3)
	pt.xticks(energies, rotation = ticks_rotation)
	if use_titles:
		pt.title("Całkowita wydajność w zależności od początkowego położenia cząstki")
	pt.ylabel("Wydajność")
	pt.xlabel("Energia fotonów [keV]")
	pt.legend(loc='upper right')
	pt.savefig("position_comparison.png", dpi = dpi, bbox_inches='tight')
	return

def resolution_comparison_histogram():
	energy = 1000
	mtx = np.array([])
	for scint_mat in scintillator_list:
		data = remove_nonpositive_elements(load_data(scint_mat,10, "G4EmLivermore", 0.01, 0, energy))
		mtx = np.concatenate((mtx, data))
		data_res = remove_nonpositive_elements(applyEnergyResolution(data, scint_mat))
		mtx = np.concatenate((mtx, data_res))
	bins = pt.hist(np.sort(mtx), bins = 100)[1]
	bin_range = np.min(mtx), np.max(mtx)
	pt.clf()
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	if use_titles:
		pt.title("Rozkład zebranych energii na scyntylatorach CeBr3 i LaBr3:Ce dla fotonów 1 MeV z i bez rozdzielczości")
	pt.ylabel("Liczba zdarzeń")
	pt.xlabel("Całkowita zebrana energia na zdarzenie [keV]")
	i = 0
	for scint_mat in scintillator_list:
		data = remove_nonpositive_elements(load_data(scint_mat,10, "G4EmLivermore", 0.01, 0, energy))
		pt.hist(data, bins = bins, histtype='step', color=colors[i], label=scint_mat + " bez rozdzielczości", range= bin_range)
		i += 1
		data_res = remove_nonpositive_elements(applyEnergyResolution(data, scint_mat))
		pt.hist(data_res, bins = bins, histtype='step', color=colors[i], label=scint_mat + " z rodzielczością", range= bin_range)
		i += 1
	ax.set_xlim(left=0)
	pt.yscale('log')
	pt.legend(loc='upper left')
	pt.savefig("resolution_comparison_histogram.png", dpi = dpi, bbox_inches='tight')

def resolution_comparison_histogram2():
	pt.clf()
	energy = 5000
	mtx = np.array([])
	simulation_indexes = range(11, 17)#range(11,17)
	srcs = len(simulation_indexes)
	mtx = np.array([])
	for index in simulation_indexes:
		data = load_data("LaBr3Ce",10, "G4EmLivermore", 0.01, index, energy)
		mtx = np.concatenate((mtx, remove_nonpositive_elements(np.sum(data, axis=1))), axis = 0)
		data_res = remove_nonpositive_elements(applyEnergyResolution(data, "LaBr3Ce"))
		mtx = np.concatenate((mtx, data_res))
	bins = pt.hist(np.sort(mtx), bins = 100)[1]
	bin_range = np.min(mtx), np.max(mtx)
	pt.clf()
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	if use_titles:
		pt.title("Rozkład zebranych energii na scyntylatorach CeBr3 i LaBr3:Ce dla fotonów 1 MeV z i bez rozdzielczości")
	pt.ylabel("Liczba zdarzeń")
	pt.xlabel("Całkowita zebrana energia na zdarzenie [keV]")
	i = 0
	for scint_mat in scintillator_list[:1]:
		data = remove_nonpositive_elements(np.sum(load_data(scint_mat,10, "G4EmLivermore", 0.01, 0, energy), axis = 1))
		print(np.shape(load_data(scint_mat,10, "G4EmLivermore", 0.01, 0, energy)))
		pt.hist(data, bins = bins, histtype='step', color=colors[i], label=scint_mat + " bez rozdzielczości", range= bin_range)
		i += 1
		data_res = remove_nonpositive_elements(applyEnergyResolution(data, scint_mat))
		pt.hist(data_res, bins = bins, histtype='step', color=colors[i], label=scint_mat + " z rodzielczością", range= bin_range)
		i += 1
		data_res = remove_nonpositive_elements(np.sum(applyEnergyResolution(load_data(scint_mat,10, "G4EmLivermore", 0.01, 0, energy), scint_mat), axis = 1))
		pt.hist(data_res, bins = bins, histtype='step', color=colors[i], label=scint_mat + " z rodzielczością sp", range= bin_range)
		i += 1
	ax.set_xlim(left=0)
	pt.yscale('log')
	pt.legend(loc='upper left')
	pt.show()
	#pt.savefig("resolution_comparison_histogram2.png", dpi = dpi,
	#bbox_inches='tight')
import numpy as np

def plot_cdf(mtx):
	sorted_data = np.sort(mtx)  # Sort the array in ascending order
	n = len(mtx)  # Number of data points
	
	# Generate an array of values from 0 to 1
	x = np.linspace(0, 1, n)
	
	# Compute the empirical cumulative distribution function
	y = np.arange(1, n + 1) / n
	
	# Plot the CDF
	pt.plot(sorted_data, y)
	
	# Set the plot labels and title
	pt.xlabel('Values')
	pt.ylabel('Cumulative Probability')
	pt.title('Cumulative Distribution Function')
	
	# Show the plot
	plt.show()

from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

def approximate_pdf(values):
	width = (np.max(values) - np.min(values)) / (10 * len(values))
	kde = gaussian_kde(values, bw_method=width)  # Perform kernel density estimation
	
	# Generate a set of values to evaluate the PDF
	x = np.linspace(np.min(values), np.max(values), 10000)
	
	# Compute the estimated PDF values
	pdf_values = kde(x)
	
	# Return the x values and corresponding PDF values
	return x, pdf_values


def find_highest_peaks(mtx_arg, n):
	
	x, mtx = approximate_pdf(mtx_arg)
	
	# Find the peaks in the distribution
	peaks, _ = find_peaks(mtx, prominence=True)
	
	# Sort the peaks based on their values
	sorted_peaks = sorted(peaks, key=lambda x: mtx[x], reverse=True)
	
	# Get the n highest peaks
	n_highest_peaks = sorted_peaks[:n]
	
	# Get the corresponding values of the n highest peaks
	highest_peak_values = mtx[n_highest_peaks]
	
	return highest_peak_values

def zero_out(array, threshold):
    array[np.isnan(array)] = 0
    array[np.isinf(array)] = 0
    array[array > threshold] = 0
    return array

def decay_anomaly_histogram():
	simulation_indexes = range(11, 17)#range(11,17)
	srcs = len(simulation_indexes)
	energy = 5000
	mtx = np.array([])
	for index in simulation_indexes:
		data = load_data("LaBr3Ce",10, "G4EmLivermore", 0.01, index, energy)
		mtx = np.concatenate((mtx, remove_nonpositive_elements(np.sum(data, axis=1))), axis = 0)
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	if use_titles:
		pt.title("Rozkład zebranych energii na scyntylatorze LaBr3:Ce dla fotonów 5 keV")
	pt.ylabel("Liczba zdarzeń")
	pt.xlabel("Całkowita zebrana energia na zdarzenie [keV]")
	pt.hist(mtx, bins = 1000, histtype='step', color=colors[0], label="LaBr3")
	ax.set_xlim(left=0)
	pt.yscale('log')
	pt.legend(loc='upper right')
	pt.savefig("decay_anomaly_histogram.png", dpi = dpi, bbox_inches='tight')

def coincidence_2d_histogram():
	energy = 583
	data = np.array([])
	gas_data = np.array([])
	for i in range(12,17):#(31, 47):
		dt, gdt = load_both("CeBr3",10, "G4EmLivermore", 0.01, i, energy, "_dataFile")
		data = np.concatenate((data, np.sum(dt, axis=1)), axis = 0)
		gas_data = np.concatenate((gas_data, np.sum(gdt,axis=1)), axis = 0)
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	if use_titles:
		pt.title("Rozkład całkowitych zebranych energii przez detektory gamma i detektor TPC")
	pt.xlabel("Całkowita zdeponowana energia na zdarzenie z rozdzielczością [keV]")
	pt.ylabel("Liczba zdarzeń")
	print(np.sum(gas_data >= 0.999 * 595) / len(gas_data) * 100)
	print(np.sum(gas_data <= 1.001 * 595) / len(gas_data) * 100)
	print(np.min(gas_data))
	plot_density_2d(data, gas_data, 100, "Całkowita energia zebrana przez detektory gamma [keV]", "Całkowita energia zebrana przez detektor TPC [keV]")
	ax.set_xlim(left=0)
	pt.savefig("coincidence_2d_histogram.png", dpi = dpi, bbox_inches='tight')
	pt.clf()
	pt.axvline(595, color='gray', linestyle='-', alpha=0.3)
	plot_density(gas_data, 1000, "Rozkład energii na gazie", "Liczba zdarzeń", "Energia zebrana na gazie [keV]")
	#pt.plot(np.sort(gas_data), np.array(range(len(gas_data)))/float(len(gas_data)))
	#pt.yscale("log")
	#pt.show()

def photopeak_coincidence_histogram():
	energy = 583
	data = np.array([])
	gas_data = np.array([])
	for i in range(7,12):#(31, 47):
		dt, gdt = load_both("CeBr3",10, "G4EmLivermore", 0.01, i, energy, "_dataFile")
		data = np.concatenate((data, np.sum(dt, axis=1)), axis = 0)
		gas_data = np.concatenate((gas_data, np.sum(gdt,axis=1)), axis = 0)
	fig, ax = pt.subplots()
	fig.set_size_inches(*inch_sizes)
	if use_titles:
		pt.title("Porównanie rozkładów energii zdeponowanej na detektorze TPC z koincydencją z detekcją fotonu gamma")
	pt.xlabel("Całkowita zdeponowana energia na zdarzenie z rozdzielczością [keV]")
	pt.ylabel("Liczba zdarzeń")
	data4, gas_data4 = remove_nonpositive_fields_both(applyEnergyResolutionCeBr3(data), applyEnergyResolutionGasOTPC(gas_data))
	gas_data_full_res_filtered = remove_nonpositive_elements(applyEnergyResolutionGasOTPC(remove_nonpositive_elements(gas_data)))
	plot_density2_same_bins(gas_data_full_res_filtered, gas_data4, 100, "Wszystkie zdarzenia w gazie", "Zdarzenia w gazie koincydujące z fotopikiem", "lower left")
	ax.set_xlim(left=0)
	pt.savefig("photopeak_coincidence_histogram.png", dpi = dpi, bbox_inches='tight')

def generate_plots():
	pt.rcParams.update({'font.size': 14})
	#cut_value_comparison()
	#physics_list_comparison()
	#scintillator_comparison()
	#detector_lenght_comparison()
	#counting_comparison()
	#groups_comparison()
	#position_comparison()
	#decay_anomaly_histogram()
	#resolution_comparison_histogram()
	#photopeak_coincidence_histogram()
	#coincidence_2d_histogram()
	#efficiency_of_photon_catching_comparison()
	detector_shadow_comparison()
	#resolution_comparison_histogram2()
	return

#matplotlib.rcParams.update({'font.size': 15})
#zad4_plt()
#this_is_where_the_fun_begins()
#test_dist()
#vis3d_TPC()
#tst()
#mif_clean()
#dist_test()
#fetch_me_plots_peasant5()
generate_plots()