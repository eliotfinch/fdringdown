This folder contains useful data associated with each interferometer. 

The text files **H1.txt**, **L1.txt**, **...** contain the posistion of each interferometer, along with the unit vectors along each arm. These are given in a coordinate system fixed at the centre of the Earth (see appendix B of <https://arxiv.org/abs/gr-qc/0008066>). In terms of latitude and longitude {phi, lambda}, the coordinate axes are oriented such that the x-axis pierces the Earth at {000, 000}, the y-axis pierces the Earth at {000, 090}, and the z-axis pierces the Earth at {090, 000}.

In the **nosie_curves** folder are various amplitude spectral densities (ASDs) associated with each interferometer. Currently available are:
 - ASDs estimated from the first three months of O3, and target ASDs for O4. These are available at <https://dcc.ligo.org/LIGO-T2000012/public>.
 - ASDs associated with the data surrounding GW150914. Available at <https://dcc.ligo.org/LIGO-P1900011/public>.
