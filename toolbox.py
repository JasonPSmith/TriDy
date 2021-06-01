# toolbox.py
# Meant for computing parameters of individual tribes of mc2 circuit
# Intended usage: 1. open python in this directory
#                 2. run exec(open('toolbox.py').read())
#                 3. call any of the functions
#                 4. compare with computed parameters by calling df.loc[index,param]

# Load packages
print('Loading packages',flush=True)
import subprocess
import time
import numpy as np
import pandas as pd
import networkx as nx
import scipy.linalg
from scipy.sparse import load_npz
from numpy.linalg import inv
import pyflagser
from pyflagsercontain import compute_cell_count
defined = {'data':{}, 'data_funcs':[], 'helper':[], 'nonspectral_params':{}, 'spectral_params':{}}

# Set working directories
dir_export = config_dict['paths']['export_address']

# Load adjacency matrix
print('Loading circuit',flush=True)
adj = load_npz(config_dict['paths']['matrix_address']).toarray().astype(int)
defined['data']['adj'] = 'adjacency matrix'


# Dictionary of all parameters which can be considered
print('Loading parameters',flush=True)
param_dict = {
    'tcc':'tcc', 'ccc':'ccc',
    'dc2':'dc2', 'dc3':'dc3', 'dc4':'dc4', 'dc5':'dc5', 'dc6':'dc6',
    'nbc':'nbc','euler_characteristic':'ec',
    'tribe_size':'tribe_size', 'degree':'deg', 'in_degree':'in_deg', 'out_degree':'out_deg', 'reciprocal_connections':'rc', 'reciprocal_connections_chief':'rc_chief',
    'asg_high':'asg', 'asg_low':'asg_low', 'asg_radius':'asg_radius',
    'tpsg_high':'tpsg', 'tpsg_low':'tpsg_low', 'tpsg_radius':'tpsg_radius',
    'tpsg_reversed_high':'tpsg_reversed', 'tpsg_reversed_low':'tpsg_reversed_low', 'tpsg_reversed_radius':'tpsg_reversed_radius',
    'clsg_low':'clsg', 'clsg_high':'clsg_high', 'clsg_radius':'clsg_radius',
    'blsg_high':'blsg', 'blsg_low':'blsg_low', 'blsg_radius':'blsg_radius',
    'blsg_reversed_high':'blsg_reversed', 'blsg_reversed_low':'blsg_reversed_low', 'blsg_reversed_radius':'blsg_reversed_radius',
}
# biedge_dict = {
#     'biedges1t': 'biedges1t', 'biedges2t': 'biedges2t', 'biedges3t': 'biedges3t', 'biedges4t': 'biedges4t', 'biedges5t': 'biedges5t', 'biedges6t': 'biedges6t',
#     'biedges1f': 'biedges1f', 'biedges2f': 'biedges2f', 'biedges3f': 'biedges3f', 'biedges4f': 'biedges4f', 'biedges5f': 'biedges5f', 'biedges6f': 'biedges6f',
#     'beulert': 'beulert', 'beulerf': 'beulerf', 'nbict': 'nbict', 'nbicf': 'nbicf'
# }
#param_dict.update(biedge_dict)
#param_dict_random = {'random_float_'+str(i).zfill(2):'randf'+str(i) for i in range(20)}
#param_dict.update(param_dict_random)
param_dict_inverse = {v: k for k, v in param_dict.items()}

# Load the parameters to be considered
param_names = config_dict['values']['selection_parameters']



print('Loading functions',flush=True)


##
## DATA FUNCTIONS
##


defined['data_funcs'].append('recompute_single(function, name, **args)')
def recompute_single(function, name, **args):
#  In: function, string
# Out: none (exports numpy array)
    data = []
    data_error = []
    for chief in range(31346):
        try:
            current_param = function(chief, **args)
            #current_param = function(tribe(chief), **args)
            data_error.append(0)
        except:
            current_param = 0
            data_error.append(1)
        data.append(current_param)
        if chief%100 == 0:
            print('Computing vertex '+str(chief)+'. So far '+str(np.count_nonzero(np.array(data_error)))+' errors.',flush=True)
    error_count = np.count_nonzero(np.array(data_error))
    print('Got '+str(error_count)+' errors ('+str(round(error_count/31346*100,2))+'%)', flush=True)

    # Check if previous array exists, backup if so
    cmd = subprocess.Popen(['ls',dir_export+'individual_parameters/'+name+'.npy'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    cmd_out = cmd.communicate()[0].decode('utf-8')
    if cmd_out != '':
        tag = round(time.time())
        print('Parameter already computed, backing up as '+dir_export+'/individual_parameters_previous/'+name+'_'+str(tag)+'.npy', flush=True)
        subprocess.run(['mv', dir_export+'individual_parameters/'+name+'.npy', dir_export+'individual_parameters_previous/'+name+'_'+str(tag)+'.npy'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('Saving params in '+dir_export+'/individual_parameters/'+name+'.npy ... ', flush=True, end='')
    np.save(dir_export+'individual_parameters/'+name+'.npy',np.array(data))
    print(' done.\nSaving error array in '+dir_export+'/individual_parameters_errors/'+name+'.npy ... ', flush=True, end='')
    np.save(dir_export+'individual_parameters_errors/'+name+'.npy',np.array(data_error,dtype='int8'))
    print(' done.',flush=True)



defined['data_funcs'].append('random_values(value_type=\'float\', value_min=0, value_max=1, name=\'random_float_00\')')
def random_values(value_type='float', value_min=0, value_max=1, name='random_float_00'):
#  In: string, float, float, string
# Out: none (exports numpy array)
    assert value_max > value_min, 'Range error: value_max must be larger than value_min'
    rng = np.random.default_rng()
    raw_data = rng.random(31346)
    if value_type == 'float':
        data = raw_data*(value_max-value_min) + value_min
    elif value_type == 'int':
        data = np.array(list(map(lambda x: int(x), data_raw)))
    else:
        print('Value error: value_type must be \'float\' or \'int\'',flush=True)
        return 0
    print('Saving params in toolbox_mc2data/individual_parameters/'+name+'.npy ... ', flush=True, end='')
    np.save(dir_export+'individual_parameters/'+name+'.npy', data)


defined['data_funcs'].append('move_top_chiefs_to_end_permutation_vector(matrix_length, chiefs_input)')
#  In: integer, list of integers
# Out: list of integers
def move_top_chiefs_to_end_permutation_vector(matrix_length, chiefs_input):
    chiefs = np.copy(chiefs_input)
    chiefs.sort()
    chiefs = chiefs[::-1]
    permutation = np.array(range(matrix_length))
    for i in range(len(chiefs)):
        swapA = permutation[chiefs[i]]
        swapB = permutation[-i-1]
        permutation[-i-1] = swapA
        permutation[chiefs[i]] = swapB
    return permutation


defined['data_funcs'].append('permute_all_but_list(matrix, list_to_fix)')
#  In: matrix, list of integers
# Out: matrix
def permute_all_but_list(matrix, list_to_fix):
    matrix_swapped = np.copy(matrix)

    # Move top chiefs to end of matrix
    matrix_length = len(matrix)
    swap_vector = move_top_chiefs_to_end_permutation_vector(matrix_length, list_to_fix)
    matrix_swapped = matrix_swapped[np.ix_(swap_vector,swap_vector)]

    # Randomly permute submatrix without top chiefs
    non_swap_length = matrix_length-len(list_to_fix)
    random_vector = np.random.permutation(non_swap_length)
    matrix_non_top_chiefs_permuted = matrix_swapped[:non_swap_length,:non_swap_length][np.ix_(random_vector,random_vector)]
    matrix_swapped[:non_swap_length,:non_swap_length] = matrix_non_top_chiefs_permuted

    # Move top chiefs back to original positions
    return_vector = np.empty_like(swap_vector)
    return_vector[swap_vector] = np.arange(swap_vector.size)
    matrix_swapped = matrix_swapped[np.ix_(return_vector,return_vector)]
    return matrix_swapped


defined['data_funcs'].append('permute_list_but_all(input_matrix, list_to_fix)')
#  In: matrix, integer
# Out: matrix
def permute_list_but_all(input_matrix, list_to_fix):
    matrix = np.copy(input_matrix)

    # Move top chiefs to end of matrix
    matrix_length = len(matrix)
    chief_length = len(list_to_fix)
    nonchief_length = matrix_length - chief_length
    swap_vector = move_top_chiefs_to_end_permutation_vector(matrix_length, list_to_fix)
    matrix = matrix[np.ix_(swap_vector,swap_vector)]

    for rowcol_index in range(chief_length):
        current_swap = np.random.permutation(nonchief_length)

        # Randomly permute columns of chiefs (outgoing neghbours)
        col_index = matrix_length-rowcol_index-1
        col = matrix[:,col_index]
        col_nochiefs = col[:nonchief_length][np.ix_(current_swap)]
        col_yeschiefs = col[nonchief_length:]
        col_new = np.hstack((col_nochiefs,col_yeschiefs))
        matrix[:,col_index] = col_new

        current_swap = np.random.permutation(nonchief_length)

        # Randomly permute rows of chiefs (incoming neghbours)
        row_index = matrix_length-rowcol_index-1
        row = matrix[row_index]
        row_nochiefs = row[:nonchief_length][np.ix_(current_swap)]
        row_yeschiefs = row[nonchief_length:]
        row_new = np.hstack((row_nochiefs,row_yeschiefs))
        matrix[row_index] = row_new

    # Move top chiefs back to original positions
    return_vector = np.empty_like(swap_vector)
    return_vector[swap_vector] = np.arange(swap_vector.size)
    matrix = matrix[np.ix_(return_vector,return_vector)]
    return matrix

#    # Permute row (incoming neighbours)
#    row = matrix[vertex]
#    random_row_nochief = row[np.arange(len(row))!=vertex][np.ix_(np.random.permutation(matrix_length-1))]
#    random_row_yeschief = np.insert(random_row_nochief,vertex,0)
#
#    # Permute col (outgoing neighbours)
#    col = np.transpose(matrix)[vertex]
#    random_col_nochief = col[np.arange(len(col))!=vertex][np.ix_(np.random.permutation(matrix_length-1))]
#    random_col_yeschief = np.insert(random_col_nochief,vertex,0)
#
#    output[vertex,:] = random_row_yeschief
#    output[:,vertex] = random_col_yeschief
#    return output


##
## HELPER FUNCTIONS (STRUCTURAL)
##

defined['helper'].append('neighbourhood(v, matrix=adj)')
def neighbourhood(v, matrix=adj):
#  In: index
# Out: list of neighbours
    neighbours = np.unique(np.concatenate((np.nonzero(matrix[v])[0],np.nonzero(np.transpose(matrix)[v])[0])))
    neighbours.sort(kind='mergesort')
    return np.concatenate((np.array([v]),neighbours))

defined['helper'].append('tribe(v)')
def tribe(v, matrix=adj):
#  In: index
# Out: adjacency matrix
    nhbd = neighbourhood(v)
    return matrix[np.ix_(nhbd,nhbd)]

defined['helper'].append('top_chiefs(parameter, number=50, order_by_ascending=False)')
def top_chiefs(parameter, number=50, order_by_ascending=False):
#  In: string, integer, boolean
# Out: list of integers
    return df.sort_values(by=[parameter],ascending=order_by_ascending)[:number].index.values

defined['helper'].append('top_nbhds(parameter, number=50, order_by_ascending=False, matrix=adj)')
def top_nbhds(parameter, number=50, order_by_ascending=False, matrix=adj):
#  In: string, integer, boolean, matrix
# Out: list of matrices
    top_chief_list = top_chiefs(parameter, number=number, order_by_ascending=order_by_ascending)
    return [neighbourhood(i, matrix=matrix) for i in top_chief_list]

defined['helper'].append('new_nhbds(nbhd_list, index_range)')
def new_nbhds(nbhd_list, index_range):
#  In: list of list of integers
# Out: list of list of integers
    new_list = []
    choice_vector = range(index_range)
    for nbhd in nbhd_list:
        new_neighbours = np.random.choice(choice_vector, size=len(nbhd)-1, replace=False)
        while nbhd[0] in new_neighbours:
            new_neighbours = np.random.choice(choice_vector, size=len(nbhd)-1, replace=False)
        new_list.append(np.hstack((nbhd[0], new_neighbours)))
    return new_list

defined['helper'].append('nx_to_np(directed_graph)')
def nx_to_np(directed_graph):
#  In: networkx directed graph
# Out: numpy array
    return nx.to_numpy_array(directed_graph,dtype=int)

defined['helper'].append('np_to_nx(adjacency_matrix)')
def np_to_nx(adjacency_matrix):
#  In: numpy array
# Out: networkx directed graph
    return nx.from_numpy_array(adjacency_matrix,create_using=nx.DiGraph)

defined['helper'].append('largest_strongly_connected_component(adjacency_matrix)')
def largest_strongly_connected_component(adjacency_matrix):
#  In: numpy array
# Out: numpy array
    current_tribe_nx = np_to_nx(adjacency_matrix)
    largest_comp = max(nx.strongly_connected_components(current_tribe_nx), key=len)
    current_tribe_strong_nx = current_tribe_nx.subgraph(largest_comp)
    current_tribe_strong = nx_to_np(current_tribe_strong_nx)
    return current_tribe_strong

defined['helper'].append('cell_count_at_v0(matrix)')
def cell_count_at_v0(matrix):
#  In: adjacency matrix
# Out: list of integers
    simplexcontainment = compute_cell_count(matrix.shape[0], np.transpose(np.array(np.nonzero(matrix))))
    return simplexcontainment[0]

defined['helper'].append('euler_characteristic_chief(chief)')
def euler_characteristic_chief(chief):
#  In: index
# Out: integer
    return euler_characteristic(tribe(chief))

defined['helper'].append('euler_characteristic(matrix)')
def euler_characteristic(matrix):
#  In: adjacency matrix
# Out: integer
    flagser_out = pyflagser.flagser_count_unweighted(matrix, directed=True)
    return sum([((-1)**i)*flagser_out[i] for i in range(len(flagser_out))])

defined['helper'].append('pyflagser.flagser_unweighted(matrix, directed=False)')


##
## HELPER FUNCTIONS (SPECTRAL)
##

defined['helper'].append('spectral_gap(matrix, thresh=10, param=\'low\')')
def spectral_gap(matrix, thresh=10, param='low'):
#  In: matrix
# Out: float
    current_spectrum = spectrum_make(matrix)
    current_spectrum = spectrum_trim_and_sort(current_spectrum, threshold_decimal=thresh)
    return spectrum_param(current_spectrum, parameter=param)

defined['helper'].append('spectrum_make(matrix)')
def spectrum_make(matrix):
#  In: matrix
# Out: list of complex floats
    assert np.any(matrix) , 'Error (eigenvalues): matrix is empty'
    eigenvalues = scipy.linalg.eigvals(matrix)
    return eigenvalues

defined['helper'].append('spectrum_trim_and_sort(spectrum, modulus=True, threshold_decimal=10)')
def spectrum_trim_and_sort(spectrum, modulus=True, threshold_decimal=10):
#  In: list of complex floats
# Out: list of unique (real or complex) floats, sorted by modulus
    if modulus:
        return np.sort(np.unique(abs(spectrum).round(decimals=threshold_decimal)))
    else:
        return np.sort(np.unique(spectrum.round(decimals=threshold_decimal)))

defined['helper'].append('spectrum_param(spectrum, parameter)')
def spectrum_param(spectrum, parameter):
#  In: list of complex floats
# Out: float
    assert len(spectrum) != 0 , 'Error (eigenvalues): no eigenvalues (spectrum is empty)'
    if parameter == 'low':
        if spectrum[0]:
            return spectrum[0]
        else:
            assert len(spectrum) > 1 , 'Error (low spectral gap): spectrum has only zeros, cannot return nonzero eigval'
            return spectrum[1]
    elif parameter == 'high':
        assert len(spectrum) > 1 , 'Error (high spectral gap): spectrum has one eigval, cannot return difference of top two'
        return spectrum[-1]-spectrum[-2]
    elif parameter == 'radius':
        return spectrum[-1]


##
## NONSPECTRAL PARAMETER FUNCTIONS
##


# transitive clustering coefficient
# source: manuscript
defined['nonspectral_params']['tcc']=[]

defined['nonspectral_params']['tcc'].append('tcc(chief_index)')
def tcc(chief_index):
    current_tribe = tribe(chief_index)
    return tcc_adjacency(current_tribe, index=chief_index)

defined['nonspectral_params']['tcc'].append('tcc_adjacency(matrix, index=0')
def tcc_adjacency(matrix, index=0):
    outdeg = np.count_nonzero(matrix[0])
    indeg = np.count_nonzero(np.transpose(matrix)[0])
    repdeg = reciprocal_connections_adjacency(matrix, chief_only=True)
    totdeg = indeg+outdeg
    chief_containment = cell_count_at_v0(matrix)
    numerator = 0 if len(chief_containment) < 3 else chief_containment[2]
    denominator = (totdeg*(totdeg-1)-(indeg*outdeg+repdeg))
    if denominator == 0:
        return 0
    return numerator/denominator


# classical clustering coefficient
# source: Clustering in Complex Directed Networks (Giorgio Fagiolo, 2006)
defined['nonspectral_params']['ccc']=[]

defined['nonspectral_params']['ccc'].append('ccc(chief_index)')
def ccc(chief_index):
    current_tribe = tribe(chief_index)
    return ccc_adjacency(current_tribe)

defined['nonspectral_params']['ccc'].append('ccc_adjacency(matrix)')
def ccc_adjacency(matrix):
    deg = degree_adjacency(matrix)
    current_size = len(matrix)
    numerator = np.linalg.matrix_power(matrix+np.transpose(matrix),3)[0][0]
    denominator = 2*(deg*(deg-1)-2*reciprocal_connections_adjacency(matrix, chief_only=True))
    if denominator == 0:
        return 0
    return numerator/denominator


# density coefficient
# source: manuscript
defined['nonspectral_params']['dc']=[]

defined['nonspectral_params']['dc'].append('dc(chief_index, coeff_index=2)')
def dc(chief_index, coeff_index=2):
#  in: index
# out: float
    current_tribe = tribe(chief_index)
    return dc_adjacency(current_tribe, chief_index=chief_index, coeff_index=coeff_index)

defined['nonspectral_params']['dc'].append('dc_adjacency(matrix, chief_index=0, coeff_index=2)')
def dc_adjacency(matrix, chief_index=0, coeff_index=2):
#  in: tribe matrix
# out: float
    assert coeff_index >= 2, 'Assertion error: Density coefficient must be at least 2'
    flagser_output = cell_count_at_v0(matrix)
    if len(flagser_output) <= coeff_index:
        density_coeff = 0
    elif flagser_output[coeff_index] == 0:
        density_coeff = 0
    else:
        numerator = coeff_index*flagser_output[coeff_index]
        denominator = (coeff_index+1)*(len(matrix)-coeff_index)*flagser_output[coeff_index-1]
        if denominator == 0:
            density_coeff = 0
        else:
            density_coeff = numerator/denominator
    return density_coeff


# normalized betti coefficient
# source: manuscript
defined['nonspectral_params']['nbc']=[]

defined['nonspectral_params']['nbc'].append('nbc(chief_index)')
def nbc(chief_index):
#  in: index
# out: float
    current_tribe = tribe(chief_index)
    return nbc_adjacency(current_tribe, chief_index=chief_index)

defined['nonspectral_params']['nbc'].append('nbc_adjacency(matrix, chief_index=0)')
def nbc_adjacency(matrix, chief_index=0):
#  in: tribe matrix
# out: float
    flagser_output = pyflagser.flagser_unweighted(matrix, directed=True)
    cells = flagser_output['cell_count']
    bettis = flagser_output['betti']
    while (cells[-1] == 0) and (len(cells) > 1):
        cells = cells[:-1]
    while (bettis[-1] == 0) and (len(bettis) > 1):
        bettis = bettis[:-1]
    normalized_betti_list = [(i+1)*bettis[i]/cells[i] for i in range(min(len(bettis),len(cells)))]
    return sum(normalized_betti_list)


# degree type parameters
defined['nonspectral_params']['degree']=[]

# tribe size
defined['nonspectral_params']['degree'].append('tribe_size(chief_index)')
def tribe_size(chief_index):
    current_tribe = tribe(chief_index)
    return tribe_size_adjacency(current_tribe)

defined['nonspectral_params']['degree'].append('tribe_size_adjacency(matrix)')
def tribe_size_adjacency(matrix):
    return len(matrix)


# degree
defined['nonspectral_params']['degree'].append('degree(chief_index, vertex_index=0)')
def degree(chief_index, vertex_index=0):
    current_tribe = tribe(chief_index)
    return degree_adjacency(matrix, vertex_index=vertex_index)

defined['nonspectral_params']['degree'].append('degree_adjacency(matrix, vertex_index=0)')
def degree_adjacency(matrix, vertex_index=0):
    return in_degree_adjacency(matrix, vertex_index=vertex_index)+out_degree_adjacency(matrix, vertex_index=vertex_index)


# in-degree
defined['nonspectral_params']['degree'].append('in_degree(chief_index, vertex_index=0)')
def in_degree(chief_index, vertex_index=0):
    current_tribe = tribe(chief_index)
    return in_degree_adjacency(current_tribe, vertex_index=vertex_index)

defined['nonspectral_params']['degree'].append('in_degree_adjacency(matrix, vertex_index=0)')
def in_degree_adjacency(matrix, vertex_index=0):
    return np.count_nonzero(matrix[vertex_index])


# out-degree
defined['nonspectral_params']['degree'].append('out_degree(chief_index, vertex_index=0)')
def out_degree(chief_index, vertex_index=0):
    current_tribe = tribe(chief_index)
    return out_degree_adjacency(current_tribe, vertex_index=vertex_index)

defined['nonspectral_params']['degree'].append('out_degree_adjacency(matrix, vertex_index=0)')
def out_degree_adjacency(matrix, vertex_index=0):
    return np.count_nonzero(np.transpose(matrix)[vertex_index])


# reciprocal connections
defined['nonspectral_params']['degree'].append('reciprocal_connections(chief_index, chief_only=False)')
def reciprocal_connections(chief_index, chief_only=False):
    current_tribe = tribe(chief_index)
    return reciprocal_connections_adjacency(current_tribe, chief_only=chief_only)

defined['nonspectral_params']['degree'].append('reciprocal_connections_adjacency(matrix, chief_only=False)')
def reciprocal_connections_adjacency(matrix, chief_only=False):
    if chief_only:
        rc_count = np.count_nonzero(np.multiply(matrix[0],np.transpose(matrix)[0]))
    else:
        rc_count = np.count_nonzero(np.multiply(matrix,np.transpose(matrix)))//2
    return rc_count


##
## SPECTRAL PARAMETER FUNCTIONS
##

# adjacency spectrum
defined['spectral_params']['asg']=[]

defined['spectral_params']['asg'].append('asg(chief_index, gap=\'high\')')
def asg(chief_index, gap='high'):
#  in: index
# out: float
    current_tribe = tribe(chief_index)
    return asg_adjacency(current_tribe, gap=gap)

defined['spectral_params']['asg'].append('asg_radius(chief_index, in_deg=False)')
def asg_radius(index, in_deg=False):
#  in: index
# out: float
    return spectrum_param(spectrum_make(tribe(index)),'radius')

defined['spectral_params']['asg'].append('asg_adjacency(matrix, gap=\'high\')')
def asg_adjacency(matrix, gap='high'):
    return spectral_gap(matrix, param=gap)


# transition probability spectrum
defined['spectral_params']['tpsg']=[]

defined['spectral_params']['tpsg'].append('tpsg(chief_index, in_deg=False, gap=\'high\')')
def tpsg(chief_index, in_deg=False, gap='high'):
#  in: index
# out: float
    current_tribe = tribe(chief_index)
    return tpsg_adjacency(current_tribe, in_deg=in_deg, gap=gap)

defined['spectral_params']['tpsg'].append('tpsg_radius(chief_index, in_deg=False)')
def tpsg_radius(index, in_deg=False):
#  in: index
# out: float
    return spectrum_param(spectrum_make(tps_matrix(tribe(index), in_deg=in_deg)),'radius')


defined['spectral_params']['tpsg'].append('tpsg_adjacency(matrix, in_deg=False, gap=\'high\')')
def tpsg_adjacency(matrix, in_deg=False, gap='high'):
#  in: tribe matrix
# out: float
    current_matrix = tps_matrix(matrix, in_deg=in_deg)
    return spectral_gap(current_matrix, param=gap)

defined['spectral_params']['tpsg'].append('tps_matrix(matrix, in_deg=False)')
def tps_matrix(matrix, in_deg=False):
#  in: tribe matrix
# out: transition probability matrix
    current_size = len(matrix)
    if in_deg:
        degree_vector = [in_degree_adjacency(matrix,vertex_index=i) for i in range(current_size)]
    else:
        degree_vector = [out_degree_adjacency(matrix,vertex_index=i) for i in range(current_size)]
    inverted_degree_vector = [0 if not d else 1/d for d in degree_vector]
    return np.matmul(np.diagflat(inverted_degree_vector),matrix)


# chung laplacian spectrum
# source 1: Laplacians and the Cheeger inequality for directed graph (Fan Chung, 2005)
# source 2: https://networkx.org/documentation/stable/reference/generated/networkx.linalg.laplacianmatrix.directed_laplacian_matrix.html
defined['spectral_params']['clsg']=[]

defined['spectral_params']['clsg'].append('clsg(chief_index, gap=\'low\')')
def clsg(chief_index, gap='low'):
#  in: index
# out: float
    current_tribe = tribe(chief_index)
    return clsg_adjacency(current_tribe, is_strongly_conn=False, gap=gap)

defined['spectral_params']['clsg'].append('clsg_radius(chief_index)')
def clsg_radius(index, gap='low'):
#  in: index
# out: float
    return spectrum_param(spectrum_make(cls_matrix_fromadjacency(tribe(index))),'radius')

defined['spectral_params']['clsg'].append('clsg_adjacency(matrix, is_strongly_conn=False, gap=\'low\')')
def clsg_adjacency(matrix, is_strongly_conn=False, gap='low'):
#  in: tribe matrix
# out: float
    chung_laplacian_matrix = cls_matrix_fromadjacency(matrix, is_strongly_conn=is_strongly_conn)
    return spectral_gap(chung_laplacian_matrix, param=gap)

defined['spectral_params']['clsg'].append('cls_matrix_fromadjacency(matrix, is_strongly_conn=False)')
def cls_matrix_fromadjacency(matrix, is_strongly_conn=False):
#  in: numpy array
# out: numpy array
    matrix_nx = np_to_nx(matrix)
    return cls_matrix_fromdigraph(matrix_nx, matrix=matrix, matrix_given=True, is_strongly_conn=is_strongly_conn)

defined['spectral_params']['clsg'].append('cls_matrix_fromdigraph(digraph, matrix=np.array([]), matrix_given=False, is_strongly_conn=False)')
def cls_matrix_fromdigraph(digraph, matrix=np.array([]), matrix_given=False, is_strongly_conn=False):
#  in: networkx digraph
# out: numpy array
    digraph_sc = digraph
    matrix_sc = matrix
    # Make sure is strongly connected
    if not is_strongly_conn:
        largest_comp = max(nx.strongly_connected_components(digraph), key=len)
        digraph_sc = digraph.subgraph(largest_comp)
        matrix_sc = nx_to_np(digraph_sc)
    elif not matrix_given:
        matrix_sc = nx_to_np(digraph_sc)
    # Degeneracy: scc has size 1
    if not np.any(matrix_sc):
        return np.array([[0]])
    # Degeneracy: scc has size 2
    elif np.array_equal(matrix_sc,np.array([[0,1],[1,0]],dtype=int)):
        return np.array([[1,-0.5],[-0.5,1]])
    # No degeneracy
    else:
        return nx.directed_laplacian_matrix(digraph_sc)


# bauer laplacian spectrum
# source: Normalized graph Laplacians for directed graphs (Frank Bauer, 2012)
defined['spectral_params']['blsg']=[]

defined['spectral_params']['blsg'].append('blsg(chief_index, reverse_flow=False, gap=\'high\')')
def blsg(chief_index, reverse_flow=False, gap='high'):
#  in: index
# out: float
    current_tribe = tribe(chief_index)
    return blsg_adjacency(current_tribe, reverse_flow=reverse_flow, gap=gap)

defined['spectral_params']['blsg'].append('blsg_radius(chief_index, reverse_flow=False)')
def blsg_radius(index, reverse_flow=False):
#  in: index
# out: float
    return spectrum_param(spectrum_make(bls_matrix(tribe(index),reverse_flow=reverse_flow)),'radius')

defined['spectral_params']['blsg'].append('blsg_adjacency(matrix, reverse_flow=False, gap=\'high\')')
def blsg_adjacency(matrix, reverse_flow=False, gap='high'):
#  in: tribe matrix
# out: float
    bauer_laplacian_matrix = bls_matrix(matrix, reverse_flow=reverse_flow)
    return spectral_gap(bauer_laplacian_matrix, param=gap)

defined['spectral_params']['blsg'].append('bls_matrix(matrix, reverse_flow=False)')
def bls_matrix(matrix, reverse_flow=False):
#  in: tribe matrix
# out: bauer laplacian matrix
    #non_quasi_isolated = [i for i in range(len(matrix)) if matrix[i].any()]
    #matrix_D = np.diagflat([np.count_nonzero(matrix[nqi]) for nqi in non_quasi_isolated])
    #matrix_W = np.diagflat([np.count_nonzero(np.transpose(matrix)[nqi]) for nqi in non_quasi_isolated])
    #return np.subtract(np.eye(len(non_quasi_isolated),dtype=int),np.matmul(inv(matrix_D),matrix_W))
    current_size = len(matrix)
    return np.subtract(np.eye(current_size,dtype='float64'),tps_matrix(matrix, in_deg=(not reverse_flow)))



##
## Load the parameters
##

# Compute the parameters if not use precomputed ones
if config_dict['values']['recompute'] == "True":
    if not os.path.exists(dir_export+'individual_parameters/'):
        print('ERROR: the folder '+dir_export+'individual_parameters/ does not exist')
    if not os.path.exists(dir_export+'individual_parameters_errors/'):
        print('ERROR: the folder '+dir_export+'individual_parameters_errors/ does not exist')
    for feature_parameter in param_names:
                    if feature_parameter == "ec":
                        recompute_single(euler_characteristic_chief, param_dict_inverse[feature_parameter])
                    elif feature_parameter == "tribe_size":
                        recompute_single(tribe_size, param_dict_inverse[feature_parameter])
                    elif feature_parameter == "deg":
                        recompute_single(degree, param_dict_inverse[feature_parameter])
                    elif feature_parameter == "in_deg":
                        recompute_single(in_degree, param_dict_inverse[feature_parameter])
                    elif feature_parameter == "out_deg":
                        recompute_single(out_degree, param_dict_inverse[feature_parameter])
                    elif feature_parameter == "rc":
                        recompute_single(reciprocal_connections, param_dict_inverse[feature_parameter])
                    elif feature_parameter == "rc_chief":
                        recompute_single(reciprocal_connections, param_dict_inverse[feature_parameter], chief_only=True)
                    elif feature_parameter == "tcc":
                        recompute_single(tcc, param_dict_inverse[feature_parameter])
                    elif feature_parameter == "ccc":
                        recompute_single(ccc, param_dict_inverse[feature_parameter])
                    elif feature_parameter[:3] == "asg":
                        if feature_parameter[4:] == "radius":
                            recompute_single(asg_radius, param_dict_inverse[feature_parameter])
                        else:
                            recompute_single(asg, param_dict_inverse[feature_parameter], gap=feature_parameter[4:])
                    elif feature_parameter[:4] == "tpsg":
                        if feature_parameter.count('_') == 2:
                            if feature_parameter[14:] == 'radius':
                                recompute_single(tpsg_radius, param_dict_inverse[feature_parameter], in_deg=True)
                            else:
                                recompute_single(tpsg, param_dict_inverse[feature_parameter], in_deg=True, gap=feature_parameter[14:])
                        else:
                            if feature_parameter[5:] == 'radius':
                                recompute_single(tpsg_radius, param_dict_inverse[feature_parameter])
                            else:
                                recompute_single(tpsg, param_dict_inverse[feature_parameter], gap=feature_parameter[5:])
                    elif feature_parameter[:4] == "clsg":
                        if feature_parameter[5:] == "radius":
                            recompute_single(clsg_radius, param_dict_inverse[feature_parameter])
                        else:
                            recompute_single(clsg, param_dict_inverse[feature_parameter], gap=feature_parameter[5:])
                    elif feature_parameter[:4] == "blsg":
                        if feature_parameter.count('_') == 2:
                            if feature_parameter[14:] == 'radius':
                                recompute_single(blsg_radius, param_dict_inverse[feature_parameter], reverse_flow=True)
                            else:
                                recompute_single(blsg, param_dict_inverse[feature_parameter], reverse_flow=True, gap=feature_parameter[14:])
                        else:
                            if feature_parameter[5:] == 'radius':
                                recompute_single(blsg_radius, param_dict_inverse[feature_parameter])
                            else:
                                recompute_single(blsg, param_dict_inverse[feature_parameter], gap=feature_parameter[5:])
                    elif feature_parameter == "nbc":
                        recompute_single(nbc, param_dict_inverse[feature_parameter])
                    elif feature_parameter[:2] == "dc":
                        recompute_single(dc, param_dict_inverse[feature_parameter], coeff_index=int(feature_parameter[2]))


# Load the computed parameters into a dataframe
param_files = [np.load(dir_export+'individual_parameters/'+param_dict_inverse[f]+'.npy') for f in param_names]
df = pd.DataFrame(np.column_stack(tuple(param_files)), columns = param_names)
defined['data']['df'] = 'mc2 parameters'

# biedge_data = {name:np.load(dir_export+'biedge_parameters/'+name+'.npy') for name in biedge_dict.keys()}
# df4 = pd.DataFrame(data = biedge_data)
# defined['data']['df4'] = 'biedge counts and derived values'
#
# df = pd.concat([df,df4],axis=1)
# defined['data']['df'] = 'mc2 and biedge parameters'



##
## PRINT AVAILABLE COMMANDS
##

if __name__ == "__main__":
    sep ='-'*32
    print(sep+'\nmc2 paramater toolbox',flush=True)
    print(sep+'\ndata',flush=True)
    for data in defined['data']:
        print(' '*(7-len(data))+data+' : '+defined['data'][data],flush=True)
    print(sep+'\ndata functions',flush=True)
    for f in defined['data_funcs']:
        print('   '+f,flush=True)
    print(sep+'\nauxiliary functions',flush=True)
    for f in defined['helper']:
        print('   '+f,flush=True)
    print(sep+'\nnon-spectral parameters',flush=True)
    for func in defined['nonspectral_params'].keys():
        print('   '+func+':',flush=True)
        for f in defined['nonspectral_params'][func]:
            print('     '+f,flush=True)
        print('\n',end='',flush=True)
    print(sep+'\nspectral parameters',flush=True)
    for func in defined['spectral_params'].keys():
        print('   '+func+':',flush=True)
        for f in defined['spectral_params'][func]:
            print('     '+f,flush=True)
        print('\n',end='',flush=True)
    print(sep,flush=True)
