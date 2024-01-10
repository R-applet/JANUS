import numpy as np
import chemprop
import numpy as np
from rdkit.Chem import *
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

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
    f_feat.write('1,0,0,0,0,0,0 \n0,1,0,0,0,0,0 \n0,0,1,0,0,0,0 \n0,0,0,1,0,0,0 \n0,0,0,0,1,0,0 \n0,0,0,0,0,1,0 \n0,0,0,0
,0,0,1')
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
    tmp = []
    for point in pareto_front:
        tmp.append(dominates(point,new_point))
    return all(not t for t in tmp)
        

def euclidean_distance(point1, point2):
    return np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(point1, point2)))

def distance_to_pareto_front(new_point, pareto_front):
    # Calculate the distance from the new point to each point in the Pareto front
    distances = [euclidean_distance(new_point, point) for point in pareto_front]
    min_dist = min(distances)
    # Return the minimum distance
    if not check_new_point(new_point, pareto_front):
        min_dist = -min_dist
    return min_dist

def dominates(point1, point2):
    # Check if point1 dominates point2
    return all(p1 >= p2 for p1, p2 in zip(point1, point2)) and any(p1 > p2 for p1, p2 in zip(point1, point2))

def identify_pareto_front(points):
    pareto_front = []
    for i, point in enumerate(points):
        if all(not dominates(other, point) for j, other in enumerate(points) if i != j):
            pareto_front.append(point)
    return pareto_front
