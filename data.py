from imports import *

URL = 'https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded'
response = requests.get(URL)

store_ds_path = ''
fname = os.path.join(store_ds_path, 'data.zip')
with open(fname, 'wb') as f:
    f.write(response.content)

with zipfile.ZipFile(fname, 'r') as f:
    f.extractall()


def loader(path):
    im = cv2.imread(path)
    return im


path = os.path.join(store_ds_path,
                    'Plant_leave_diseases_dataset_without_augmentation')

tfms = transforms.Compose([
    lambda x: cv2.resize(x, (150, 150)),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.4, contrast=0, saturation=0, hue=0),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(
        degrees=(0, 360),
        translate=(0.1, 0.1),
        scale=(0.5, 1)
    )
])

ds = datasets.ImageFolder(path, loader=loader, transform=tfms)

def beautify_labels(label):
    bucket = re.findall(r'[^,_() ]+', label)[:3]
    bucket = list(map(lambda x: x.capitalize(), bucket))
    return ''.join(bucket)

label_names = list(map(beautify_labels, ds.classes))

label2idx = {key: val for val, key in enumerate(label_names)}
idx2label = {val: key for key, val in label2idx.items()}

train_ds, val_ds, test_ds  = torch.utils.data.random_split(ds, (45448, 5000, 5000))