
# coding: utf-8

# # The NRTL Thermodynamic Model

# The NRTL is an activity coefficient model.  For a species $i$ in a mixture of $n$ components, the activity coefficient $\gamma_i$ is given by:
# $$
#     \ln{\gamma_i} = \frac{\sum_{j=1}^n{x_j\tau_{ji}G_{ji}}}{\sum_{k=1}^n{x_kG_{ki}}} + \sum_{j=1}^n{\frac{x_jG_{ij}}{\sum_{k=1}^n{x_kG_{kj}}}}\left(\tau_{ij} - \frac{\sum_{m=1}^n{x_mG_{mj}\tau_{mj}}}{\sum_{k=1}^n{x_kG_{kj}}}\right)
# $$
# where,<br>
# $G_{ij} = \exp{\left(-\alpha_{ij}\tau_{ij}\right)}$ <br>
# $\tau_{ij} = a_{ij} + \frac{b_{ij}}{T} + c_{ij}\ln{T} + d_{ij}T$ ... Extended Antoine Format: only useful if temperature different is greater than 50 K. <br>
# $\alpha_{ij}$ is a non-randomness parameter which is 0 for a random mixture and 0.2 for regular solutions and 0.48 for highly ordered solutions (like highly polar or hydrogen bonded solutions). <br>
# 
# Generally, $\alpha_{ij} = \alpha_{ji}$ but $\tau_{ij} \neq \tau_{ji}$.<br>
# <br>
# For two components, $n=2$, this reduces to:
# $$
#     \ln{\gamma_1} = \frac{x_1\tau_{11}G_{11} + x_2\tau_{21}G_{21}}{x_1G_{11}+x_2G_{21}} + \frac{x_1G_{11}}{x_1G_{11}+x_2G_{21}}\left(\tau_{11} - \frac{x_1\tau_{11}G_{11}+x_2\tau_{21}G_{21}}{x_1G_{11}+x_2G_{21}}\right) + \frac{x_2G_{12}}{x_1G_{12}+x_2G_{22}}\left(\tau_{12} - \frac{x_1\tau_{12}G_{12} + x_2\tau_{22}G_{22}}{x_1G_{12}+x_2G_{22}} \right)
# $$
# Since $\tau_{11} = \tau_{22} = 0$ and $G_{11}=G_{22}=1$,
# 

# $$
#     \ln{\gamma_1} = \frac{x_2\tau_{21}G_{21}}{x_1+x_2G_{21}} - \frac{x_1}{x_1+x_2G_{21}}\frac{x_2\tau_{21}G_{21}}{x_1+x_2G_{21}} + \frac{x_2G_{12}}{x_1G_{12}+x_2}\tau_{12} - \frac{x_2G_{12}}{x_1G_{12}+x_2}\frac{x_1\tau_{12}G_{12}}{x_1G_{12}+x_2}
# $$

# i.e.
# $$
#     \ln{\gamma_1} = \frac{x_1x_2\tau_{21}G_{21}+x_2^2\tau_{21}G_{21}^2}{\left(x_1+x_2G_{21}\right)^2} - \frac{x_1x_2\tau_{21}G_{21}}{\left(x_1+x_2G_{21}\right)^2} + \frac{x_2x_1\tau_{12}G_{12}^2+x_2^2\tau_{12}G_{12}}{\left(x_1G_{12}+x_2\right)^2} - \frac{x_2x_1\tau_{12}G_{12}^2}{\left(x_1G_{12}+x_2\right)^2}
# $$

# i.e.
# $$
#     \ln{\gamma_1} = \frac{x_2^2\tau_{21}G_{21}^2}{\left(x_1+x_2G_{21}\right)^2} + \frac{x_2^2\tau_{12}G_{12}}{\left(x_1G_{12}+x_2\right)^2} = x_2^2\left(\frac{\tau_{21}G_{21}^2}{\left(x_1+x_2G_{21}\right)^2} + \frac{\tau_{12}G_{12}}{\left(x_1G_{12}+x_2\right)^2}\right)
# $$

# And from symmetry we have:
# $$
#     \ln{\gamma_2} =  x_1^2\left(\frac{\tau_{12}G_{12}^2}{\left(x_2+x_1G_{12}\right)^2} + \frac{\tau_{21}G_{21}}{\left(x_2G_{21}+x_1\right)^2}\right)
# $$

# ## NRTL object class
# We can now define an NRTL object class.  The class takes in a number of species and their names, converts the names into indices (using dictionaries) and generates matrices of {$a_{ij}$}, {$b_{ij}$}, {$c_{ij}$}, {$d_{ij}$} and {$\alpha_{ij}$}.  It will have a method to populate the matrices by taking in dictionaries of pairwise interactions.  Then it will have another method to generate the $\gamma_i$ values for a given $T$ and a dictionary of mol-fractions.  <br>
# The dictionary of pairwise interactions for, say matrix C, is as follows: <br>
# <code>dict_parameters_C = {"water-ipa", [-0.0021, 0.0036], "water-meoh",[-0.235, 0.225]}</code> where the first is $C_{water-ipa}$ and the second is $C_{ipa-water}$. <br>
# To get $\gamma_i$ values, a dictionary of mol-fractions is passed to the method <code>get_gammas</code>. 
# 

# $$
#     \ln{\gamma_i} = \frac{\sum_{j=1}^n{x_j\tau_{ji}G_{ji}}}{\sum_{k=1}^n{x_kG_{ki}}} + \left(\sum_{j=1}^n{\tau_{ij}G_{ij}\frac{x_j}{\sum_{k=1}^n{x_kG_{kj}}}} - \sum_{j=1}^n{G_{ij}\frac{x_j}{\sum_{k=1}^n{x_kG_{kj}}}}\frac{\sum_{m=1}^n{x_mG_{mj}\tau_{mj}}}{\sum_{k=1}^n{x_kG_{kj}}}\right)
# $$
# 

# Let the mole fractions be a column vector $\mathbf{x}$, the {$\tau_{ij}$} form a tensor $\textbf{T}$ while the {$G_{ij}$} form a tensor $\textbf{G}$. The matrix {$\tau_{ij}G_{ij}$} is $\textbf{TG}$.
# Hence, 
# $$
#     xTG = \sum_{j=1}^n{x_j\tau_{ji}G_{ji}} = \textbf{x}^T \cdot \textbf{TG} 
# $$
# This is an $1 \times n$ row vector.
# $$
#     xG = \sum_{j=1}^n{x_jG_{ji}} = \textbf{x}^T \cdot \textbf{G}
# $$
# This is also an $1 \times n$ row vector. <br>
# Hence the vector $xTG\_xG = \left[\frac{\sum_{j=1}^n{x_j\tau_{ji}G_{ji}}}{\sum_{j=1}^n{x_jG_{ji}}}\right]$ is also an $1 \times n$ row vector.
# The vector $x\_xG = \left[\frac{x_j}{\sum_{k=1}^n{x_kG_{kj}}}\right]$ is $\textbf{x_xG}$ an $n \times 1$ column vector.
# Hence,
# $$
#     TGx\_xG = \sum_{j=1}^n{\tau_{ij}G_{ij}\frac{x_j}{\sum_{k=1}^n{x_kG_{kj}}}} = \textbf{TG}\cdot \textbf{x_xG}
# $$
# This is an $n \times 1$ column vector. <br>
# Also, $xxTG\_xG2 =\left[x_j\frac{\sum_{m=1}^n{x_mG_{mj}\tau_{mj}}}{\left(\sum_{k=1}^n{x_kG_{kj}}\right)^2}\right]$ is $\textbf{xxTG_xG2}$ a $n \times 1$ column vector. 
# Hence:
# $$
#     GxxTG\_xG2 = \sum_{j=1}^n{G_{ij}x_j\frac{\sum_{m=1}^n{x_mG_{mj}\tau_{mj}}}{\left(\sum_{k=1}^n{x_kG_{kj}}\right)^2}} = \textbf{G}\cdot \textbf{xxTG_xG2}
# $$
# This is an $n \times 1$ column vector.
# 

# Hence:
# $$
#     \mathbf{\vec{\ln{\gamma}}} = xTG\_xG^T + TGx\_xG - GxxTG\_xG2
# $$

# In[3]:

import scipy
import scipy.linalg


# In[4]:

class NRTL:
    def __init__(self, list_of_components):  #list of components are names of the components
        self.list_of_components = list_of_components
        self.get_matrices(self.list_of_components)
    def get_indices(self, list_of_components):
        self.list_of_components = list_of_components
        self.dict_index_component = {i:name for (i,name) in zip(range(len(list_of_components)), list_of_components)}
        self.dict_component_index = {name:i for (i,name) in zip(range(len(list_of_components)), list_of_components)}
    def get_matrices(self, list_of_components):
        self.get_indices(list_of_components)
        n = len(list_of_components)
        A = scipy.zeros((n,n))
        B = scipy.zeros((n,n))
        C = scipy.zeros((n,n))
        D = scipy.zeros((n,n))
        alpha = scipy.zeros((n,n))
        self.dict_matrices = {"A":A, "B":B, "C":C, "D":D, "alpha":alpha}
    def populate_matrix(self, matrix_name, dict_parameters_matrix):
        '''
        populate_matrix(matrix_name, dict_parameters_matrix)
        e.g. 
        dict_parameters_matrix = {"water-ipa", [-0.0021, 0.0036], "water-meoh",[-0.235, 0.225]}
        matrix_name should be in ['A','B','C','D','alpha']
        '''
        for key in dict_parameters_matrix:
            comp1, comp2 = key.split('-')
            index1, index2 = self.dict_component_index[comp1], self.dict_component_index[comp2]
            self.dict_matrices[matrix_name][index1, index2] = dict_parameters_matrix[key][0]
            self.dict_matrices[matrix_name][index2, index1] = dict_parameters_matrix[key][1]
    def yield_parameters(self):
        '''
        Returns a dictionary with the matrix names ['A','B','C','D','alpha'] as keys.  
        Values are other dictionaries with component pairs as keys and the corresponding [1-2, 2-1] interactions tuple as values.
        '''
        dict_parameters = {
            "A":{},
            "B":{},
            "C":{},
            "D":{},
            "alpha":{}
        }
        for i in range(len(self.list_of_components)):
            name_i = self.dict_index_component[i]
            for j in range(len(self.list_of_components)):
                if j > i:
                    name_j = self.dict_index_component[j]
                    key = name_i+'-'+name_j
                    dict_parameters["A"][key] = [self.dict_matrices["A"][i,j], self.dict_matrices["A"][j,i]]
                    dict_parameters["B"][key] = [self.dict_matrices["B"][i,j], self.dict_matrices["B"][j,i]]
                    dict_parameters["C"][key] = [self.dict_matrices["C"][i,j], self.dict_matrices["C"][j,i]]
                    dict_parameters["D"][key] = [self.dict_matrices["D"][i,j], self.dict_matrices["D"][j,i]]
                    dict_parameters["alpha"][key] = [self.dict_matrices["alpha"][i,j], self.dict_matrices["alpha"][j,i]]
        return dict_parameters
    def get_gammas(self, T, dict_mol_fractions):
        n = len(self.list_of_components)
        x = scipy.zeros((len(self.list_of_components),1)) # n x 1 matrix (column vector)
        for key in dict_mol_fractions:
            index = self.dict_component_index[key]
            x[index,0] = dict_mol_fractions[key]
        
        x = x/sum(x) #normalizing x.  n x 1 column vector
        
        Tau = self.dict_matrices["A"] + self.dict_matrices["B"]/T + self.dict_matrices["C"]*scipy.log(T) + self.dict_matrices["D"]*T
        G = scipy.exp(-self.dict_matrices["alpha"]*Tau) #n x n matrices
        TG = Tau*G #n x n matrix
        
        xTG = scipy.dot(x.T, TG).T # n x 1 column vector
        xG  = scipy.dot(x.T, G).T  # n x 1 column vector
        xTG_xG = xTG/xG          # n x 1 column vector
        x_xG = x/xG              # n x 1 column vector 
        TGx_xG = scipy.dot(TG, x_xG) #n x 1 column vector
        xxTG_xG2 = x*xTG/(xG*xG) #n x 1 column vector
        GxxTG_xG2 = scipy.dot(G, xxTG_xG2) #n x 1 column vector
        
        matrix_log_gamma = xTG_xG + TGx_xG - GxxTG_xG2 #n x 1 column vector
        
        array_log_gamma = matrix_log_gamma.reshape((n,))
        array_gamma = scipy.exp(array_log_gamma)
        dict_gamma = {name:gamma for (name, gamma) in zip([self.dict_index_component[index] for index in range(n)],array_gamma)}
        return dict_gamma


# nrtl = NRTL(["water", "ipa"])

# nrtl.dict_component_index

# nrtl.dict_index_component

# dict_params = {"water-ipa":[-0.0023, 0.0035]}
# nrtl.populate_matrix("A", dict_params)

# nrtl.dict_matrices["A"]

# dict_params = {"water-ipa":[0.3, 0.3]}
# nrtl.populate_matrix("alpha", dict_params)

# nrtl.populate_matrix("D",{"ipa-water":[5.0,6.0]})
# nrtl.dict_matrices["D"]

# x = scipy.zeros((4,1))
# x[2,0] = 3
# x[0,0] = 4
# x[1,0] = 6
# x[3,0] = 8

# x = x/sum(x)

# x.reshape((4,))

# x.reshape((1,4))
# x[0,0]

# nrtl.get_gammas(300.0, {"water":0.5, "ipa":0.001})

# 0.2*1.00069024*(-0.0023)+0.8*1*0

# nrtl.yield_parameters()

# 
