import pandas as pd
import chemprop
import numpy as np
from rdkit.Chem import *
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from scipy.interpolate import interp1d
import pickle
import os

def make_preds(smi: str, model_path: str, scale_path: str, col_names: list, gen: int):
    scale_dict = pickle.load(open(scale_path,'rb'))
    props = list(scale_dict.keys())
    smiles = MolToSmiles(MolFromSmiles(smi))
    
    os.mkdir('./tmp')
    
    f_smi = open('tmp/smi_tmp.txt','w')
    f_smi.write('smiles\n')
    f_smi.write(f'{smiles}\n')
    f_smi.close()

    arguments = [
        '--test_path', 'tmp/smi_tmp.txt',
        '--preds_path', 'tmp/pred_tmp.csv',
        '--checkpoint_dir', model_path
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)

    df_pred = pd.read_csv('tmp/pred_tmp.csv')

    p = []
    for i in range(len(props)):
       p.append(scale_dict[props[i]].inverse_transform(df_pred[col_names[i]][0].reshape(-1,1))[0][0]) 

    os.system('rm -r tmp')

    return p

def make_preds_selector(smi: str, model_path: str, scale_path: str, gen: int):
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
        '--preds_path', 'tmp/pred_tmp.csv',
        '--checkpoint_dir', model_path
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)

    df_pred = pd.read_csv('tmp/pred_tmp.csv')

    p = []
    for i in range(len(df_pred)):
        p.append(list(scale_dict.values())[i].inverse_transform(df_pred['scaled_val'][i].reshape(-1,1))[0][0])

    os.system('rm -r tmp')
    #record_data(smi, p, gen)

    return p

def collect_ensemble(smi: str, model_paths: str, scale_paths: str, col_names: list, gen: int):
    ps = []
    for i in range(len(model_paths)):
        p_i = make_preds(smi,model_paths[i],scale_paths[i],col_names,gen)
        ps.append(p_i)
    ps_array = np.array(ps)
    p_means = []
    p_std = []
    for j in range(len(ps_array[0])):
        p_means.append(np.mean(ps_array[:,j]))
        p_std.append(np.std(ps_array[:,j]))
    
    record_data(smi, p_means, p_std, gen)
    return p_means,p_std

def check_new_point(new_point, pareto_front, opt):
    dominated = False
    for point in pareto_front:
        if opt == 'min_max':
            if (point[0] < new_point[0] and point[1] > new_point[1]):
                dominated = True
                break

        elif opt == 'max_min':
            if (point[0] > new_point[0] and point[1] < new_point[1]):  # other is better in both objectives
                dominated = True
                break

        elif opt == 'min_min':
            if (point[0] < new_point[0] and point[1] < new_point[1]):  # other is better in both objectives
                dominated = True
                break

        elif opt == 'max_max':
            if (point[0] > new_point[0] and point[1] > new_point[1]):  # other is better in both objectives
                dominated = True
                break
        
    return dominated
        
def euclidean_distance(point1, point2):
    return np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(point1, point2)))

def distance_to_pareto_front(new_point, pareto_fit, opt):
    # Calculate the distance from the new point to each point in the Pareto front
    distances = [euclidean_distance(new_point, point) for point in pareto_fit]
    min_dist = min(distances)
    # Return the minimum distance
    if check_new_point(new_point, pareto_fit, opt):
        min_dist = -min_dist
    return min_dist

def find_pareto_front(costs,opt):
    pareto_front = []
    for i, c in enumerate(costs):
        # Check if point 'c' is dominated by any other point
        dominated = False
        for other in costs:
            if opt == 'min_max':
                if (other[0] < c[0] and other[1] > c[1]):  # other is better in both objectives
                    dominated = True
                    break

            elif opt == 'max_min':
                if (other[0] > c[0] and other[1] < c[1]):  # other is better in both objectives
                    dominated = True
                    break

            elif opt == 'min_min':
                if (other[0] < c[0] and other[1] < c[1]):  # other is better in both objectives
                    dominated = True
                    break

            elif opt == 'max_max':
                if (other[0] > c[0] and other[1] > c[1]):  # other is better in both objectives
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
    
    x_fit = np.linspace(min(x_data), max(x_data), 20000)

    
    # Fit the curve using interp1D
    fitter = interp1d(x_data, y_data)
    y_fit = fitter(x_fit)
    
    pf_data = []
    for i in range(len(x_fit)):
        pf_data.append([x_fit[i],y_fit[i]])
    return np.array(pf_data)

def fit_step(points, density, opt):
    segments = []
    if opt == 'min_max':
        sorted_points = np.sort(points,axis=0)
        for i in range(len(points)):
            if i == 0:
                x_up = np.linspace(sorted_points[i][0],sorted_points[i][0],density)
                y_up = np.linspace(0,sorted_points[i][1],density)
                x_right = np.linspace(sorted_points[i][0],sorted_points[i+1][0],density)
                y_right = np.linspace(sorted_points[i][1],sorted_points[i][1],density)

            else:
                x_up = np.linspace(sorted_points[i][0],sorted_points[i][0],density)
                y_up = np.linspace(sorted_points[i-1][1],sorted_points[i][1],density)
                x_right = np.linspace(sorted_points[i-1][0],sorted_points[i][0],density)
                y_right = np.linspace(sorted_points[i-1][1],sorted_points[i-1][1],density)

            xy_up = np.vstack([x_up,y_up]).T
            xy_right = np.vstack([x_right,y_right]).T
            segments.append(xy_up)
            segments.append(xy_right)

        x_right = np.linspace(sorted_points[-1][0],1000,density)
        y_right = np.linspace(sorted_points[-1][1],sorted_points[-1][1],density)
        xy_right = np.vstack([x_right,y_right]).T
        segments.append(xy_right) 

    elif opt == 'max_min':
        sorted_points = points[points[:, 0].argsort()[::-1]]
        for i in range(len(points)):
            if i == 0:
                x_down = np.linspace(sorted_points[i][0],sorted_points[i][0],density)
                y_down = np.linspace(1000,sorted_points[i][1],density)
                x_left = np.linspace(sorted_points[i][0],sorted_points[i+1][0],density)
                y_left = np.linspace(sorted_points[i][1],sorted_points[i][1],density)

            else:
                x_down = np.linspace(sorted_points[i][0],sorted_points[i][0],density)
                y_down = np.linspace(sorted_points[i-1][1],sorted_points[i][1],density)
                if i == len(points)-1:
                    x_left = np.linspace(sorted_points[i][0],-100,density)
                    y_left = np.linspace(sorted_points[i][1],sorted_points[i][1],density)
                else:
                    x_left = np.linspace(sorted_points[i][0],sorted_points[i+1][0],density)
                    y_left = np.linspace(sorted_points[i][1],sorted_points[i][1],density)

            xy_down = np.vstack([x_down,y_down]).T
            xy_left = np.vstack([x_left,y_left]).T
            segments.append(xy_down)
            segments.append(xy_left)

    elif opt == 'min_min':
        sorted_points = points[points[:, 0].argsort()[::-1]]
        for i in range(len(points)):
            if i == 0:
                x_up = np.linspace(sorted_points[i][0],sorted_points[i][0],density)
                y_up = np.linspace(sorted_points[i][1],sorted_points[i+1][1],density)
                x_left = np.linspace(1000,sorted_points[i][0],density)
                y_left = np.linspace(sorted_points[i][1],sorted_points[i][1],density)

            else:
                x_left = np.linspace(sorted_points[i-1][0],sorted_points[i][0],density)
                y_left = np.linspace(sorted_points[i][1],sorted_points[i][1],density)
                if i == len(points)-1:
                    x_up = np.linspace(sorted_points[i][0],sorted_points[i][0],density)
                    y_up = np.linspace(sorted_points[i][1],1000,density)
                else:
                    x_up = np.linspace(sorted_points[i][0],sorted_points[i][0],density)
                    y_up = np.linspace(sorted_points[i][1],sorted_points[i+1][1],density)

            xy_up = np.vstack([x_up,y_up]).T
            xy_left = np.vstack([x_left,y_left]).T
            segments.append(xy_up)
            segments.append(xy_left)

    elif opt == 'max_max':
        sorted_points = points[points[:, 0].argsort()[::-1]]
        for i in range(len(points)):
            if i == 0:
                x_up = np.linspace(sorted_points[i][0],sorted_points[i][0],density)
                y_up = np.linspace(0,sorted_points[i][1],density)
                x_left = np.linspace(sorted_points[i][0],sorted_points[i+1][0],density)
                y_left = np.linspace(sorted_points[i][1],sorted_points[i][1],density)

            else:
                x_up = np.linspace(sorted_points[i][0],sorted_points[i][0],density)
                y_up = np.linspace(sorted_points[i-1][1],sorted_points[i][1],density)
                if i == len(points)-1:
                    x_left = np.linspace(sorted_points[i][0],-100,density)
                    y_left = np.linspace(sorted_points[i][1],sorted_points[i][1],density)
                else:
                    x_left = np.linspace(sorted_points[i][0],sorted_points[i+1][0],density)
                    y_left = np.linspace(sorted_points[i][1],sorted_points[i][1],density)

            xy_up = np.vstack([x_up,y_up]).T
            xy_left = np.vstack([x_left,y_left]).T
            segments.append(xy_up)
            segments.append(xy_left)

    return np.vstack(segments)

def fit_step_old(points, density):
    sorted_points = np.sort(points,axis=0)
    segments = []
    for i in range(len(points)):
        
        if i == 0:
            x_up = np.linspace(sorted_points[i][0],sorted_points[i][0],density)
            y_up = np.linspace(0,sorted_points[i][1],density)
            x_right = np.linspace(sorted_points[i][0],sorted_points[i+1][0],density)
            y_right = np.linspace(sorted_points[i][1],sorted_points[i][1],density)

        else:
            x_up = np.linspace(sorted_points[i][0],sorted_points[i][0],density)
            y_up = np.linspace(sorted_points[i-1][1],sorted_points[i][1],density)
            x_right = np.linspace(sorted_points[i-1][0],sorted_points[i][0],density)
            y_right = np.linspace(sorted_points[i-1][1],sorted_points[i-1][1],density)

        xy_up = np.vstack([x_up,y_up]).T
        xy_right = np.vstack([x_right,y_right]).T
        segments.append(xy_up)
        segments.append(xy_right)

    x_right = np.linspace(sorted_points[-1][0],1000,density)
    y_right = np.linspace(sorted_points[-1][1],sorted_points[-1][1],density)
    xy_right = np.vstack([x_right,y_right]).T
    segments.append(xy_right)        
    return np.vstack(segments)

def record_data(smi: str, props: list, stds: list, col_names: list, gen: int):
    add_line = smi
    for p in props:
        add_line+=f',{p}'
    for s in stds:
        add_line+=f',{s}'
    exists = os.path.exists('master.txt')
    if not exists:
        f = open('master.txt','a')
        tmp_str = 'smiles'
        for name in col_names:
            tmp_str += ','+name.split('_')[0]
        tmp_str += ',generation\n'
        f.write(tmp_str)
        f.write(add_line+f',{gen}\n')
        f.close()
    else:
        master = pd.read_csv('master.txt')
        if smi in master['smiles'].to_list():
            pass
        else:
            f = open('master.txt','a')
            f.write(add_line+f',{gen}\n')
            f.close()
