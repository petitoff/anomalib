from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

datamodule = Folder(
    name="wzor_pasek_A",
    root="./dataset/wzor_pasek_A",
    normal_dir="train/good",
    abnormal_dir="test/defect",
    normal_test_dir="test/good",
    normal_split_ratio=0.2,
    extensions=[".PNG"],
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8
)

model = Patchcore(
    backbone="wide_resnet50_2",        
    layers=["layer2", "layer3"],       
    coreset_sampling_ratio=0.1         
)

engine = Engine()
engine.fit(model=model, datamodule=datamodule)

