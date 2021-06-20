import numpy as np
import matplotlib.pyplot as plt

C_E = [0.0, 0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2.0]

L_valid_mean = [1.952127759, 0.0005962159034,
                0.0004711394664, 0.0005834673447,
                0.0002941389565, 0.0003296977101,
                0.0004494255529, 0.002946071369]
L_valid_median = [0.5921077357, 0.0003066186729,
                    0.0003341875426, 0.0004471611268,
                    0.0002345053829, 0.0002968166058,
                    0.000219119246, 0.002882682313]

L_E_valid_mean = [1.948738322, 0.00003620224218,
                0.00002932581798, 0.00009763408846,
                0.00001551633784, 0.00001807109486,
                0.0000282040731, 0.0005843581096]

L_E_valid_median = [0.5880123939, 0.00001674534104,
                    0.00001309517456, 0.00004713261921,
                    0.00001261580431, 0.000006486507691,
                    0.00001253029425, 0.0005758334238]

L_D_valid_mean = [0.003389437674, 0.0005600136613,
                0.0004418136484, 0.0004858332563,
                0.0002786226186, 0.0003116266153,
                0.0004212214798, 0.00236171326]

L_D_valid_median = [0.002697785816, 0.0002976488903,
                    0.0003275666461, 0.000387200132,
                    0.0002281029478, 0.0002930149164,
                    0.0001838607645, 0.002117200133]


L_train_mean = [1.874220708, 0.000021322535725,
                0.00002593112838,0.0002121868538,
                0.00002749324442, 0.00004344412544,
                0.0004132141182, 0.006393353206]
L_train_median = [0.543109262, 0.00002151229839,
                    0.0000283708513, 0.00002974429627,
                    0.00002687580328, 0.00003313690565,
                    0.00004861559875, 0.00593153324]

L_E_train_mean = [1.873808446, 0.0000007664321582,
                0.0000001395877072, 0.00002514130497,
                0.00000002149509181, 0.0000005545773536,
                0.000001317800144, 1.25E-15]

L_E_train_median = [0.5431044275, 0.0000001177518524,
                    0.00000003580178262, 0.00000003287279752,
                    0.000000006661519034, 0.000000002130819912,
                    0.000000009995367976, 6.10E-16]

L_D_train_mean = [0.0004123533986, 0.000020556115,
                0.00002579156145, 0.0001870455931,
                0.00002747176656, 0.00004288953569,
                0.0004118963147, 0.006228352173]

L_D_train_median = [0.000006154424582, 0.00002113862619,
                    0.00002820370331, 0.00002963110332,
                    0.00002686073612, 0.00003313298175,
                    0.00004858311194, 0.005931533181]

plt.plot(C_E[1:-1], L_valid_mean[1:-1],'o',label='Mean validation cost',ls='--')
plt.plot(C_E[1:-1], L_train_mean[1:-1],'o',label='Mean training cost',ls='--')
plt.plot(C_E[1:-1], L_valid_median[1:-1],'o',label='Median validation cost',ls='--',color='g')
plt.plot(C_E[1:-1], L_train_median[1:-1],'o',label='Median training cost',ls='--',color='r')
plt.xlabel("$C_E$")
plt.ylabel("Loss")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()
plt.savefig('L_means.png')
plt.show()


plt.plot(C_E[1:-1], L_E_valid_mean[1:-1],'o',label='Mean validation cost',ls='--')
plt.plot(C_E[1:-1], L_E_train_mean[1:-1],'o',label='Mean training cost',ls='--')
plt.plot(C_E[1:-1], L_E_valid_median[1:-1],'o',label='Median validation cost',ls='--',color='g')
plt.plot(C_E[1:-1], L_E_train_median[1:-1],'o',label='Median training cost',ls='--',color='r')
plt.xlabel("$C_E$")
plt.ylabel("Loss")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()
plt.savefig('L_E_means.png')
plt.show()


plt.plot(C_E[1:-1], L_D_valid_mean[1:-1],'o',label='Mean validation cost',ls='--')
plt.plot(C_E[1:-1], L_D_train_mean[1:-1],'o',label='Mean training cost',ls='--')
plt.plot(C_E[1:-1], L_D_valid_median[1:-1],'o',label='Median validation cost',ls='--',color='g')
plt.plot(C_E[1:-1], L_D_train_median[1:-1],'o',label='Median training cost',ls='--',color='r')
plt.xlabel("$C_E$")
plt.ylabel("Loss")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()
plt.savefig('L_D_means.png')
plt.show()
