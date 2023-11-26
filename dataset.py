import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


HEIGHT = 300
WIDTH = 400
RESIZE_WIDTH = 1066
RESIZE_HEIGHT = 800
PAD_HEIGHT_SIZE = 800
PAD_WIDTH_SIZE = 1088

H_PAD = int((PAD_HEIGHT_SIZE - RESIZE_HEIGHT) /2)
W_PAD = int((PAD_WIDTH_SIZE - RESIZE_WIDTH) /2)
    
def overlay_boxes_on_image(image, label, bbox, iterator):
    """Visualize bounding boxes overlayed on images."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    color_dict = {1: "blue", 2: "green", 3: "red"}
    label_dict = {1: "Vehicle", 2: "Person", 3: "Animal"}
    
    ax.imshow(image)

    # Iterate through bounding boxes and labels
    for index, coordinates in enumerate(bbox):
        startX, startY, endX, endY = coordinates

        # Compute the width and height of the bounding box (# need to verify this)
        box_width = endX - startX
        box_height = endY - startY

        box = patches.Rectangle((startX, startY), box_width, box_height, 
                                linewidth=3, edgecolor=color_dict[label[index]], 
                                facecolor='none')

        # A box style to be used when displaying the text label (# can we use this method? check !)
        text_box_props = dict(boxstyle='round', facecolor=color_dict[label[index]], alpha=0.5)

        ax.text(startX, startY - 4, label_dict[label[index]], 
                fontsize=10, color="white", 
                verticalalignment='top', 
                bbox=text_box_props)
        ax.add_patch(box)
    
    plt.savefig("testfig/visualtrainset" + str(iterator) + ".png")
    plt.show()


def apply_alpha_channel(image_channel, mask, alpha, intensity):
    """Adjust the alpha channel on the image based on the mask and intensity."""
    return np.where(
        mask > 0,
        image_channel * (1 - alpha) + alpha * intensity,
        image_channel
    )


def process_each_channel(image, mask, label, alpha, channel_mapping):
    """Process each channel of the image and apply the mask accordingly."""
    modified_channels = []
    
    for channel_index in range(3):
        intensity = 1 if channel_index == channel_mapping[label] else 0
        modified_channel = apply_alpha_channel(image[:, :, channel_index], mask, alpha, intensity)
        modified_channels.append(modified_channel)
    
    return np.stack(modified_channels, axis=-1)


def overlay_mask_on_image(image, masks, labels, alpha=0.4):
    """Overlay masks on the given image with a specified transparency level."""
    modified_image = image.copy()

    channel_mapping = {1: 2, 2: 1, 3: 0}

    for index in range(len(masks)):
        single_mask = masks[index]
        label = labels[index]
        modified_image = process_each_channel(modified_image, single_mask, label, alpha, channel_mapping)
        
    return modified_image

def normalize_numpyarray_plot(np_array):
    """Normalize a NumPy array to the range [0, 1] for plotting."""
    array_min, array_max = np.min(np_array), np.max(np_array)
    safe_denominator = max(array_max - array_min, np.finfo(float).eps)
    normalized_array = (np_array - array_min) / safe_denominator
    return normalized_array



class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        # TODO: load dataset, make mask list
        self.bboxes = np.load(paths[3],allow_pickle=True, encoding='latin1')
        self.labels = np.load(paths[2], allow_pickle=True, encoding='latin1')
        self.images = np.array(h5py.File(paths[0], 'r+')['data']).astype("uint8").transpose((0,2,3,1))
        self.masks = np.array(h5py.File(paths[1], 'r+')['data']).astype("int8")
        self.masks = np.expand_dims(self.masks,-1)
        self.masks = self.unflatten_masks(self.masks, self.bboxes)
        
        ### preprocessing of datas

        self.transform = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Resize((RESIZE_HEIGHT,RESIZE_WIDTH)),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.Pad([W_PAD,H_PAD])
        ])

        self.mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((RESIZE_HEIGHT,RESIZE_WIDTH)),
        transforms.Pad([W_PAD,H_PAD])
        ])

    def scale_pad_y(self, y_coord):
        y_coord = y_coord * (RESIZE_HEIGHT/HEIGHT) + H_PAD
        return y_coord

    def scale_pad_x(self, x_coord):
        x_coord = x_coord * (RESIZE_WIDTH/WIDTH) + W_PAD
        return x_coord

    def unflatten_masks(self, masks, bboxes):
        indices = np.cumsum([0] + [len(bbox) for bbox in bboxes])
        batched_masks = [masks[start:end] for start, end in zip(indices[:-1], indices[1:])]
        return np.array(batched_masks, dtype=object)
    
    def process_masks(self, masks):
        processed = [torch.squeeze(self.mask_transform(m)) for m in masks]
        return torch.stack(processed)

    def transed_bbox(self, bboxes):
        return np.array([
            [
                self.scale_pad_x(box[0]),
                self.scale_pad_y(box[1]),
                self.scale_pad_x(box[2]),
                self.scale_pad_y(box[3])
            ]
            for box in bboxes
        ])

    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
    def __getitem__(self, index):
        # TODO: __getitem__
  
        image = self.images[index]
        bbox = self.bboxes[index]
        label = self.labels[index]
        mask = self.masks[index]

        if self.transform:
            image = self.transform(image)

        mask = self.process_masks(mask) if self.mask_transform else mask
        mod_bbox = self.transed_bbox(bbox.copy())

        assert image.shape == (3, 800, 1088)
        assert mod_bbox.shape[0] == mask.shape[0]

        return image, label, mask, mod_bbox
        
    # check flag
    # return transed_img, label, transed_mask, transed_bbox

    def __len__(self):
        return len(self.images)

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    # def pre_process_batch(self, img, mask, bbox):
    #     # TODO: image preprocess

    #     # check flag
    #     assert img.shape == (3, 800, 1088)
    #     assert bbox.shape[0] == mask.squeeze(0).shape[0]
    #     return img, mask, bbox


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        images, labels, masks, bounding_boxes = list(zip(*batch))
        return torch.stack(images), labels, masks, bounding_boxes

    def loader(self):
        return DataLoader(self.dataset, batch_size= self.batch_size, collate_fn=self.collect_fn)

# Below Code Is Present in cis6800_hw3.ipynb
# ## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path = '/content/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '/content/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = '/content/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = '/content/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)
    print("Loaded the data into torch.utils.data.Dataset")

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    print(f"Train size: {train_size} & Test Size: {test_size}")
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")

    # Create the directory if it doesn't exist
    output_dir = 'testfig'
    os.makedirs(output_dir, exist_ok=True)
    
    for iter, data in enumerate(train_loader, 0):
        img, label, mask, bbox = [data[i] for i in range(len(data))]
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        label = [label_img for label_img in label]
        mask = [mask_img for mask_img in mask]
        bbox = [bbox_img for bbox_img in bbox]

        for idx in range(len(img)):
            plot_image = normalize_numpyarray_plot(img[idx].permute((1, 2, 0)).cpu().numpy())
            masked_image = overlay_mask_on_image(plot_image, mask[idx], label[idx], alpha=0.5)
            overlay_boxes_on_image(masked_image, label[idx], bbox[idx])

        if iter == 10:
            break