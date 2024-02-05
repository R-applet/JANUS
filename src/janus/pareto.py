import pandas as pd
import chemprop
import numpy as np
from rdkit.Chem import *
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from scipy.interpolate import Akima1DInterpolator
import pickle
import os

def make_preds(smi: str, model_path: str, scale_path: str):
    scale_dict = pickle.load(open(scale_path,'rb'))
    nprops = len(scale_dict.keys())

    smiles = MolToSmiles(MolFromSmiles(smi))

    os.mkdir('./tmp')

    f_smi = open('tmp/smi_tmp.txt','w')
    f_smi.write('smiles\n')
    for i in range(nprops):
        f_smi.write(f'{smiles}\n')
    f_smi.close()

    f_feat = open('tmp/feats_tmp.csv','w')
    f_feat.write('s1,s2,s3,s4,s5,s6,s7 \n')
    f_feat.write('1,0,0,0,0,0,0 \n0,1,0,0,0,0,0 \n0,0,1,0,0,0,0 \n0,0,0,1,0,0,0 \n0,0,0,0,1,0,0 \n0,0,0,0,0,1,0 \n0,0,0,0,0,0,1')
    f_feat.close()

    arguments = [
        '--test_path', 'tmp/smi_tmp.txt',
        '--features_path', 'tmp/feats_tmp.csv',
        '--no_features_scaling',
        '--ensemble_variance',
        '--preds_path', 'tmp/pred_tmp.csv',
        '--checkpoint_dir', model_path
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)

    df_pred = pd.read_csv('tmp/pred_tmp.csv')

    p = []
    for i in range(len(df_pred)):
        p.append(list(scale_dict.values())[i].inverse_transform(df_pred['val'][i].reshape(-1,1))[0][0])

    os.system('rm -r tmp')
    record_data(smi, p)

    return p

def check_new_point(new_point, pareto_front):
    dominated = False
    for point in pareto_front:
        if (point[0] < new_point[0] and point[1] > new_point[1]):
            dominated = True
            break
    return dominated
        
def euclidean_distance(point1, point2):
    return np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(point1, point2)))

def distance_to_pareto_front(new_point, pareto_fit):
    # Calculate the distance from the new point to each point in the Pareto front
    distances = [euclidean_distance(new_point, point) for point in pareto_front]
    min_dist = min(distances)
    # Return the minimum distance
    if check_new_point(new_point, pareto_fit):
        min_dist = -min_dist
    return min_dist

def find_pareto_front(costs):
    pareto_front = []
    for i, c in enumerate(costs):
        # Check if point 'c' is dominated by any other point
        dominated = False
        for other in costs:
            if (other[0] < c[0] and other[1] > c[1]):  # other is better in both objectives
                dominated = True
                break
        if not dominated:
            pareto_front.append(c)
    return np.array(pareto_front)


def fit_curve_to_points(points):
    # Sort the points based on x-values
    sorted_points = sorted(points, key=lambda point: point[0])
    
    x_data, y_data = zip(*sorted_points)
    
    # Convert to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    x_fit = np.linspace(min(x_data), max(x_data), 5000)

    
    # Fit the curve using Akima 
    fitter = Akima1DInterpolator(x_data, y_data)
    y_fit = fitter.__call__(x_fit)
    
    pf_data = []
    for i in range(len(x_fit)):
        pf_data.append((x_fit[i],y_fit[i]))
    return pf_data

def record_data(smi: str, props: list):
    add_line = smi
    for p in props:
        add_line+=f',{p}'
    exists = os.path.exists('master.txt')
    if not exists:
        f = open('master.txt','a')
        f.write('smiles,mpC,Tdec,density_exp,density_calc,hof_calc,log(h50),log(E50)\n')
        f.close()

    f = open('master.txt','a')
    f.write(add_line+'\n')
    f.close()
