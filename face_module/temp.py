import torch
from settings import DIR
from pathlib import Path
import sys
#sys.path.insert(0, './trained_model')

FAS = torch.load(Path(DIR.joinpath('face_module/trained_model/model_checkpoint/FAS.pt')), map_location=torch.device('cpu'))