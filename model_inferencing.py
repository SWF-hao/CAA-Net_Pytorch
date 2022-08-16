
import argparse
import logging
import torch
import torch.nn as nn
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.IK_dataset import IK_Dataset
import csv



# Data preprocessing
data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def eval_model(dataloaders,model_name,csv_name):
    ff = open("outputs/"+csv_name + '.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(ff)
    csv_writer.writerow(["img name", "Normal","VK","FK","BK", "class_idx", "class_name"])
    model = torch.load(model_name,map_location=device)
    model.eval()
        # Iterate over data.
    for batch in tqdm(dataloaders):
        
        name = batch["name"]
        inputs = batch["image"].to(device)

        softmax = nn.Softmax(dim=1)
        outputs,cls_out = model(inputs)
        results = softmax(outputs) + cls_out
        results = softmax(results)
        _, preds = torch.max(results, 1)
        for idx in range(len(results)):
            res=results.cpu().detach().numpy().tolist()
            class_idx = res[idx].index(max(res[idx]))
            class_list = ["Normal","VK","FK","BK"]
            csv_writer.writerow([name[idx],res[idx][0],res[idx][1],res[idx][2],res[idx][3],class_idx,class_list[class_idx]])


    ff.close()





def get_args():
    parser = argparse.ArgumentParser(description='Inference parameters of CAA-Net using slit-lamp images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', '--weight', metavar='W', type=str, default="CAA-Net.pt",
                        help='Weight of inference model', dest='weight')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-i', '--input-images', metavar='I', type=str, nargs='?', default="examples",
                        help='Directory of input images', dest='I')
    parser.add_argument('-d', '--device', dest='device', type=str, default="cpu",
                        help='Device:0,1,2,3....or cpu')
    parser.add_argument('-o', '--output-name', dest='o', type=str, default="examples",
                        help='The output .csv name')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    dataset = IK_Dataset(image_dir="inputs/"+args.I, transform=data_transforms)
    dataloaders = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    dataset_sizes = len(dataset)
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" +args.device if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device {device}\n'
                 f'Weight location weights/{args.weight}\n'
                 f'Batch size {args.batchsize}\n'
                 f'Input dir {args.I}\n'
                 f'Output name outputs/{args.o}.csv\n')
    eval_model(dataloaders, "weights/"+args.weight,args.o)
    print("Inferencing result saved at outputs/"+args.o+".csv")

