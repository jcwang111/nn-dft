import numpy as np
import matplotlib.pyplot as plt

C_E = [0.5, 0.7, 1.0,
       1.2, 1.5, 1.7]

L_test_mean = [0.0001212153213, 0.0001220685735, 0.0004564361129,
               0.0002166104438, 0.000311422409, 0.0006230390246]

L_test_median = [0.0001098002123, 0.0000961397825, 0.0001366018946,
                 0.000174125873, 0.0002777081222, 0.0002386674573]

L_E_test_mean = [0.000008019173869, 0.00000628676054, 0.00001596706289,
                 0.000008264940245, 0.00002092730123, 0.0001698216481]

L_E_test_median = [0.000006693258373, 0.000005451548894, 0.000004104696702,
                   0.000007416711544, 0.00001645159461, 0.00001975577742]

L_D_test_mean = [0.0001131961474, 0.000115781813, 0.00044046905,
                 0.0002083455035, 0.0002904951078, 0.0004532173765]

L_D_test_median = [0.0001062032006, 0.00008891863003, 0.0001319480917,
                   0.0001659644054, 0.0002624278821, 0.0002277192333]


plt.plot(C_E, L_test_mean,'o',label='Mean test loss',ls='--',color='orange')
plt.plot(C_E, L_test_median,'o',label='Median test loss',ls='--',color='g')

plt.xlabel("$C_E$")
plt.ylabel("Total Loss")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()
plt.savefig('L_means.png')
plt.show()


plt.plot(C_E, L_E_test_mean,'o',label='Mean test energy loss',ls='--')
plt.plot(C_E, L_E_test_median,'o',label='Median test energy loss',ls='--',color='g')
plt.xlabel("$C_E$")
plt.ylabel("$L_E$")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()
plt.savefig('L_E_means.png')
plt.show()


plt.plot(C_E, L_D_test_mean,'o',label='Mean test density loss',ls='--')
plt.plot(C_E, L_D_test_median,'o',label='Median test density loss',ls='--',color='g')
plt.xlabel("$C_E$")
plt.ylabel("$L_D$")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()
plt.savefig('L_D_means.png')
plt.show()
