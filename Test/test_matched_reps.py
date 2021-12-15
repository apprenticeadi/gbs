from _loop_hafnian_subroutines import matched_reps
from loop_hafnian import loop_hafnian
import numpy as np


C = np.array([[0.13417546, 0.14196733, 0.1999522 , 0.0042831 ],
       [0.14196733, 0.02651059, 0.13954111, 0.17235905],
       [0.1999522 , 0.13954111, 0.01813927, 0.12274551],
       [0.0042831 , 0.17235905, 0.12274551, 0.18099043]])

G = np.array([0.05712311, 0.03520913, 0.48682174, 0.59668519])

pattern = np.array([2,0,1,1])

H = loop_hafnian(C, G, pattern, False)

