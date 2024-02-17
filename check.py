import numpy as np
from scipy import stats
from scipy.stats import wilcoxon
# lenet5
a = np.array([0.315, 0.7638, 0.3563, 0.4758, 0.4146, 0.4504, 0.2489, 0.201, 0.5474, 0.6272, 0.5815, 0.3381, 0.2638, 0.4565, 0.4504, 0.6938, 0.5452, 0.2693, 0.4609, 0.4504])
b = np.array([0.8756]*20)
s,p = stats.shapiro(a)
print(s)
print(p)  # p>0.5服从正态分布

w, p_value_w = wilcoxon(a, b)
print(p_value_w)

print([0.87]*20)



