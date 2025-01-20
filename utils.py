import cv2
import gc
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models import mobilenet_v2, efficientnet_b0, densenet121, shufflenet_v2_x1_0

from models import *

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.1):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes, smoothing=0.1):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size()[0], n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                         self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


loss_fun = SmoothCrossEntropyLoss(smoothing=0.1)

class TrafficSignDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, device):
        self.x = x
        self.y = torch.LongTensor(y)
        self.device = device

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        _x = transforms.ToTensor()(self.x[item]).to(self.device)
        _y = self.y[item].to(self.device)
        return _x, _y

    
    # Preprocess images (code works with numpy arrays in the augmentation part -> needs to be nomalized) (we skipp shifting to [-0.5,0.5])
def pre_process_image(image):

    # Normalize to values between 0 and 1
    image = image / 255.
    return image


# Augmentation (extra training data)
def transform_image(image, ang_range, shear_range, trans_range, preprocess):

    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = image.shape
    rot_m = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    shear_m = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, rot_m, (cols, rows))
    image = cv2.warpAffine(image, trans_m, (cols, rows))
    image = cv2.warpAffine(image, shear_m, (cols, rows))

    image = pre_process_image(image) if preprocess else image

    return image


# Generates extra training data
def gen_extra_data(x_train, y_train, n_each, ang_range, shear_range, trans_range, randomize_var, preprocess=True):

    if os.path.exists('dataset/GTSRB/x_arr.npy') and os.path.exists('dataset/GTSRB/y_arr.npy'):
        print("Loading saved data...")
        x_arr = np.load('dataset/GTSRB/x_arr.npy')
        y_arr = np.load('dataset/GTSRB/y_arr.npy')
    else:
        print("Generating data...")
        x_arr, y_arr = [], []
        n_train = len(x_train)

        pbar = tqdm(total=n_train, desc="Processing Images", ncols=80)

        for i in range(n_train):
            for i_n in range(n_each):
                img_trf = transform_image(x_train[i],
                                        ang_range, shear_range, trans_range,
                                        preprocess)
                x_arr.append(img_trf)
                y_arr.append(y_train[i])
            pbar.update()

        pbar.close()

        print('for loop done')

        x_arr = np.array(x_arr, dtype=np.float32())

        print('xarr done')

        y_arr = np.array(y_arr, dtype=np.float32())

        print('yarr done')

        if randomize_var == 1:
            len_arr = np.arange(len(y_arr))
            print('np arange done')
            np.random.shuffle(len_arr)
            print('shuffle done')
            x_arr[len_arr] = x_arr
            print('first inplace done')
            y_arr[len_arr] = y_arr
            print('second inplace done')

        np.save('dataset/GTSRB/x_arr.npy', x_arr)
        np.save('dataset/GTSRB/y_arr.npy', y_arr)

        # Save data
        print("Saving data to disk...")

    return x_arr, y_arr

def weights_init(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.05)
        nn.init.constant_(m.bias, 0.05)


def model_epoch(training_model, data_loader, train=False, optimizer=None, device=None):

    loss = acc = 0.0
    total_inference_time = 0.0  # variable to keep track of the total inference time

    for data_batch in data_loader:
        # Record start time
        start_time = time.time()

        train_predict = training_model(data_batch[0].to(device))

        # Record end time
        end_time = time.time()

        # Calculate inference time for this batch and add it to the total
        total_inference_time += end_time - start_time

        batch_loss = loss_fun(train_predict, data_batch[1].to(device))
        if train:
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        acc += (torch.argmax(train_predict, dim=1) == data_batch[1].to(device)).sum()
        loss += batch_loss.item() * len(data_batch[1])

    return acc, loss, total_inference_time


def training(training_model, train_data, train_labels, test_data, test_labels, model_name, device):

    best_acc = 0.0

    num_epoch, batch_size = 100, 64
    optimizer = torch.optim.Adam(
        training_model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(num_epoch):

        extra_train, extra_labels = (train_data, train_labels)

        train_set = TrafficSignDataset(extra_train, extra_labels, device)
        test_set = TrafficSignDataset(test_data, test_labels, device)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        epoch_start_time = time.time()

        training_model.train()
        train_acc, train_loss, _ = model_epoch(
            training_model, train_loader, train=True, optimizer=optimizer, device=device)

        training_model.eval()
        with torch.no_grad():
            test_acc, test_loss, _ = model_epoch(training_model, test_loader, device=device)

        actual_acc = test_acc

        # Update best weights
        if actual_acc > best_acc:
            torch.save(training_model.state_dict(),
                    f'models/{model_name}.pth')
            best_acc = actual_acc

        print(f'[{epoch+1}/{num_epoch}] {round(time.time() - epoch_start_time, 2)}', end=' ')
        print(f'Train Acc: {round(float(train_acc / train_set.__len__()), 4)}', end=' ')
        print(f'Loss: {round(float(train_loss / train_set.__len__()), 4)}', end=' | ')
        print(f'Test Acc: {round(float(test_acc / test_set.__len__()), 4)}', end=' ')
        print(f'Loss: {round(float(test_loss / test_set.__len__()), 4)}')

        del extra_train, extra_labels, train_set, train_loader
        gc.collect()
        
        
# Load model from path
def load_model(params):

    # Get parameters from params dictionary
    path_model = params['PATH_MODEL']
    class_n = params['CLASS_N']
    model_type = params['MODEL_TYPE']
    device = params['DEVICE']

    # Load model architecture
    if model_type=='CNNsmall':
        trained_model = CNNsmall(class_n=class_n).to(device)
    elif model_type=='CNNlarge':
        trained_model = CNNlarge(class_n=class_n).to(device)
    elif model_type=='Transformer':
        trained_model = Transformer(class_n=class_n).to(device)
    elif model_type=='ResNet18':
        trained_model = ResNet18(class_n=class_n).to(device) 
    elif model_type=='MobileNetv2':
        trained_model = mobilenet_v2(num_classes=class_n).to(device) 
    elif model_type=='EfficientNet':
        trained_model = efficientnet_b0(num_classes=class_n).to(device) 
    elif model_type=='DenseNet121':
        trained_model = densenet121(num_classes=class_n).to(device)
    elif model_type=='ShuffleNetv2x1':
        trained_model = shufflenet_v2_x1_0(num_classes=class_n).to(device) 

    # Load weights
    trained_model.load_state_dict(
        torch.load(path_model,
                   map_location=torch.device(device)))

    # Set to evaluation mode
    trained_model.eval()

    # Retun model
    return trained_model


# Test model on frame
def test_on_frame(model, frame, labels=None):
    # Frame to tensor
    transform_frame = transforms.Compose([transforms.Resize((32, 32)),
                                         transforms.ToTensor()])
    # Preprocess the image and prepare it for prediction.
    frame = transform_frame(frame)
    predict = torch.softmax(model(frame)[0], 0)
    index = int(torch.argmax(predict).data)
    confidence = float(predict[index].data)
    if labels is None:
      print(f'Predict: {index} (class index) Confidence: {confidence*100}%')
    else:
      print(f'Predict: {index} ({labels[index]}) Confidence: {confidence*100}%')


# Test model on frame from path
def test_on_frame_from_path(model, path_frame, labels=None):

    # Frame to tensor
    transform_frame = transforms.Compose([transforms.Resize((32, 32)),
                                         transforms.ToTensor()])

    # Preprocess the image and prepare it for prediction.
    frame = torch.clamp(transform_frame(Image.open(path_frame)), 0, 1)
    #img_processed = img_processed.unsqueeze(0).to(device)
    predict = torch.softmax(model(frame)[0], 0)
    index = int(torch.argmax(predict).data)
    confidence = float(predict[index].data)
    if labels is None:
      print(f'Predict: {index} (class index) Confidence: {confidence*100}%')
    else:
      print(f'Predict: {index} ({labels[index]}) Confidence: {confidence*100}%')



def test_on_tensor(model, tensor, params, labels=None, specific_class=None):
    tensor = tensor.unsqueeze(0).to(params['DEVICE'])
    predict = torch.softmax(model(tensor)[0], 0)
    index = int(torch.argmax(predict).data)
#     print("All predictions:", predict)
#     print("Second largest prediction:", torch.topk(predict, 2))
    print("Class with the max confidence is ", labels[index])
    confidence = float(predict[index].data)
    if labels is None:
        print(f'Predict: {index} (class index) Confidence: {confidence*100}%')
    else:
        print(f'Predict: {index} ({labels[index]}) Confidence: {confidence*100}%')
    if specific_class:
        confidence = float(predict[specific_class].data)
        print(f'Predict: {specific_class} ({labels[specific_class]}) Confidence: {confidence*100}%')


# Loads model, sign, and sign mask
def load_attack_toolset(params):

    # Load model
    model = load_model(params)

    # Sign to tensor
    transform_sign = transforms.Compose([transforms.Resize((32, 32)),
                                         transforms.ToTensor()])

    # Load original sign as tensor in required size
    sign = transform_sign(Image.open(params['PATH_SIGN']))

    # Noise to size of sign and to tensor
    transform_noise = transforms.Compose([transforms.Resize((32, 32)),
                                         transforms.ToTensor()])

    # Load mask of sign as tensor in required size
    sign_mask = transform_noise(Image.open(params['PATH_SIGN_MASK']).convert('RGB'))

    # Threshold the mask (sometimes the mask is not exactly zero or 1)
    sign_mask = (sign_mask > 0.5).float()

    return model, sign, sign_mask



# Stores a tensor as an image
def store_tensor_as_image(filename, tensor):
    # Convert the final perturbed image (with noise but without augmentation) to a PIL Image and save it as a PNG
    tensor_array = tensor.detach().cpu().numpy() * 255
    if tensor_array.shape[0] == 1:  # Grayscale image, squeeze the channel dimension
        tensor_array = np.squeeze(tensor_array, axis=0)
    else:  # Color image, rearrange dimensions to (height, width, channels)
        tensor_array = np.transpose(tensor_array, (1, 2, 0))
    final_image = Image.fromarray(tensor_array.astype(np.uint8))
    final_image.save(filename)


def store_tensor_as_tensor(filename, tensor):
    torch.save(tensor.detach().cpu(), filename)


# Loads an image as a PIL image
def load_image_as_tensor(path):
    return Image.open(path)


# Differentiable augmentations to make the attacks more robust
def differentiable_augment(image, augment, max_rotation=15, max_translation=1, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):

    image = image.unsqueeze(0)  # Add the batch dimension if it's not present
    B, C, H, W = image.shape

    if augment:
        output = image

        # Randomly adjust brightness, contrast, saturation, and hue
        output = TF.adjust_brightness(output, 1 + (torch.rand(1).item() - 0.5) * brightness)
        output = TF.adjust_contrast(output, 1 + (torch.rand(1).item() - 0.5) * contrast)
        output = TF.adjust_saturation(output, 1 + (torch.rand(1).item() - 0.5) * saturation)
        output = TF.adjust_hue(output, (torch.rand(1).item() - 0.5) * hue)

        # Randomly rotate the image
        angle = (torch.rand(1) - 0.5) * 2 * max_rotation  # Generate a random angle between -max_rotation and max_rotation
        output = TF.rotate(output, angle.item())  # Apply the rotation
    else:
        output = image

    return output.squeeze(0)  # Remove the batch dimension before returning


# Converts a tensor to a mask (all channels of a pixel are one or zero)
def tensor_to_mask(tensor, threshold):
    # Create mask where any channel value is larger than the threshold
    mask = (torch.abs(tensor) > threshold).float()

    mask = (mask == 1).any(dim=0)

    # Erweitern der Dimensionen der Maske, um die Tensorgröße anzupassen
    mask = mask.unsqueeze(0).expand_as(tensor)

    # Setzen der Pixel, wo die Maske True ist, auf 1, ansonsten auf 0
    tensor = torch.where(mask, torch.ones_like(tensor), torch.zeros_like(tensor))

    return tensor


# Train a mask for a given target class
def create_noise_mask(model, stop_sign_image, sign_mask, params):

    # Ensure the model is in evaluation mode
    model.eval()

    # Move the images and masks to the same device as the model
    stop_sign_image = stop_sign_image.to(params['DEVICE'])
    print("image", stop_sign_image.shape)
    sign_mask = sign_mask.to(params['DEVICE'])
    print("sign_mask", sign_mask.shape)

    # Create a tensor for the noise, initialized to zero
    noise = torch.zeros_like(stop_sign_image, requires_grad=True).to(params['DEVICE'])

    beta1 = params['INIT_MASK_BETA_1']
    beta2 = params['INIT_MASK_BETA_2']
    beta3 = params['INIT_MASK_BETA_3']

    # Create an optimizer for the noise
    optimizer = optim.Adam([noise], lr=params['INIT_MASK_LEARNING_RATE'])

    for i in range(params['INIT_MASK_EPOCHS']):
        optimizer.zero_grad()

        # Add the noise to the stop sign image, and clip the values to be between 0 and 1
        perturbed_image = torch.clamp(stop_sign_image + (noise * sign_mask), 0, 1)

        # Augment the perturbed image
        augmented_image = differentiable_augment(perturbed_image.clone(), augment=params['INIT_MASK_AUGMENTATION'])

        # Pass the augmented image through the model
        output = model(augmented_image.unsqueeze(0).to(params['DEVICE']))

        # Calculate the loss
        loss = beta3 * F.cross_entropy(output, torch.tensor([params['TARGET_CLASS']]).to(params['DEVICE']))

        # Calculate differences between channels
        diff1 = torch.abs(noise[0, :, :] - noise[1, :, :])
        diff2 = torch.abs(noise[0, :, :] - noise[2, :, :])

        # Apply some function to penalize differences. In this case, we use square to emphasize larger differences.
        color_penalty = torch.sum(diff1**2 + diff2**2)

        # Add color_penalty to the loss
        loss += beta1 * color_penalty

        # Add regularization (l1 norm of noise)
        loss += beta2 * torch.norm(noise, 1)

        # Backpropagate the gradients
        loss.backward()

        # Update the noise
        optimizer.step()

        # Print the loss every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}, Loss = {loss.item()}")
            plt.imshow(augmented_image.permute(1, 2, 0).detach().cpu().numpy())
            plt.show()

    # Convert the final perturbed image (with noise but without augmentation) to a PIL Image and save it as a PNG
    # store_tensor_as_image('noise.png', noise)

    return tensor_to_mask(noise, params['INIT_MASK_THRESHOLD'])



# Train a perturbed sign as adversarial attack
def train_attack(params):

  model, stop_sign_image, sign_mask = load_attack_toolset(params)

  # Create a mask to limit noise to a defined area
  noise_mask = create_noise_mask(model, stop_sign_image, sign_mask, params)

  # Print mask proposal
  print('This is the mask we suggest:')
  plt.imshow(noise_mask.permute(1, 2, 0).detach().cpu().numpy())
  plt.show()

  # Ensure the model is in evaluation mode
  model.eval()

  # Move the images and masks to the same device as the model
  stop_sign_image = stop_sign_image.to(params['DEVICE'])
  noise_mask = noise_mask.to(params['DEVICE'])

  # Create a tensor for the noise, initialized to zero
  noise = torch.zeros_like(stop_sign_image, requires_grad=True).to(params['DEVICE'])

  beta1 = params['ATTACK_BETA_1']
  beta2 = params['ATTACK_BETA_2']
  beta3 = params['ATTACK_BETA_3']

  # Create an optimizer for the noise
  optimizer = optim.Adam([noise], lr=params['ATTACK_LEARNING_RATE'])

  for i in range(params['ATTACK_EPOCHS']):

      optimizer.zero_grad()

      # Add the noise to the stop sign image, and clip the values to be between 0 and 1
      perturbed_image = torch.clamp(stop_sign_image + (noise * noise_mask), 0, 1)

      # Initialize loss of one batch
      loss = 0

      # Calculate the loss for a batch
      for j in range(params['ATTACK_BATCH_SIZE']):

        # Augment the perturbed image
        augmented_image = differentiable_augment(perturbed_image.clone(), augment=params['ATTACK_AUGMENTATION'])

        # Pass the augmented image through the model
        output = model(augmented_image.unsqueeze(0).to(params['DEVICE']))

        # Create target output tensor
        # target_output = torch.zeros_like(output).to(params['DEVICE'])
        # target_output[0][params['TARGET_CLASS']] = 1

        # Calculate the loss
        # loss += beta3 * F.binary_cross_entropy_with_logits(output, target_output)

        # Alternative loss calculation
        loss +=  beta3 * F.cross_entropy(output, torch.tensor([params['TARGET_CLASS']]).to(params['DEVICE']))

        # Calculate differences between channels
        diff1 = torch.abs(noise[0, :, :] - noise[1, :, :])
        diff2 = torch.abs(noise[0, :, :] - noise[2, :, :])

        # Apply some function to penalize differences. In this case, we use square to emphasize larger differences.
        color_penalty = torch.sum(diff1**2 + diff2**2)

        # Add color_penalty to the loss
        loss += beta1 * color_penalty

        # Add regularization (l2 norm of noise)
        loss += beta2 * torch.norm(noise, 2)

      # Backpropagate the gradients
      loss.backward()

      # Update the noise
      optimizer.step()

      # Print the loss every 100 iterations
      if i % 100 == 0:
          print(f"Iteration {i}, Loss = {loss.item()}")
          plt.imshow(augmented_image.permute(1, 2, 0).detach().cpu().numpy())
          plt.show()

  # Convert the final perturbed image (with noise but without augmentation) to a PIL Image and save it as a PNG
  store_tensor_as_image(params['PATH_NOISE_SMALL'], noise)
  store_tensor_as_tensor(params['PATH_NOISE_TENSOR'], noise)
  store_tensor_as_image(params['PATH_NOISE_MASK_SMALL'], noise_mask)
  store_tensor_as_image(params['PATH_PERT_SIGN_SMALL'], torch.clamp(stop_sign_image + (noise * noise_mask), 0, 1))
  store_tensor_as_image(params['PATH_AUGMENTED_SIGN_SMALL'], augmented_image)
  store_tensor_as_image(params['PATH_ORIG_SIGN_SMALL'], stop_sign_image)

  # Classify original stop sign with attack
  test_on_tensor(tensor=torch.clamp(stop_sign_image + (noise * noise_mask), 0, 1), model=model, params=params, labels=params['LABELS'])


# Plots the final pertubated sign and the classification result
def test_attack(params):

    # Transform sign to tensor
    transform_sign = transforms.Compose([transforms.ToTensor()])  # TODO: transforms.Lambda(lambda x: x - 0.5)

    # Load sign and transform to tensor
    sign = transform_sign(Image.open(params['PATH_SIGN']))

    # Transform noise to tensor and resize to sign size
    transform_noise_mask = transforms.Compose([transforms.Resize((sign.shape[1], sign.shape[2])),
                                          transforms.ToTensor()])

    noise_mask = torch.clamp(transform_noise_mask(Image.open(params['PATH_NOISE_MASK_SMALL'])), 0, 1)

    noise = torch.load(params['PATH_NOISE_TENSOR'])

    # Ensure the tensor is 4D, with the batch dimension first and the channel dimension second
    if len(noise.shape) == 3:
        noise = noise.unsqueeze(0)  # add a batch dimension

    # Now we can resize
    noise = F.interpolate(noise, size=(sign.shape[1], sign.shape[2]), mode='bilinear', align_corners=False)

    # If you want to remove the batch dimension afterward
    noise = noise.squeeze(0)

    output = torch.clamp(sign + noise * noise_mask, 0, 1)

    store_tensor_as_image(params['PATH_PERT_SIGN_LARGE'], output)
    store_tensor_as_image(params['PATH_NOISE_LARGE'], noise)
    store_tensor_as_image(params['PATH_NOISE_MASK_LARGE'], noise_mask)

    # Visualize the tensor
    plt.imshow(output.permute(1, 2, 0), cmap='gray')
    plt.colorbar()
    plt.show()

    transform_to_tensor = transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.ToTensor()])

    pert_sign = torch.clamp(transform_to_tensor(Image.open(params['PATH_PERT_SIGN_LARGE'])), 0, 1)
    orig_sign = torch.clamp(transform_to_tensor(Image.open(params['PATH_SIGN'])), 0, 1)

    model = load_model(params)
    
    print("Prediction on the perturbed sign:")
    test_on_tensor(model=model, tensor=pert_sign, params=params, labels=params['LABELS'])
    test_on_tensor(model=model, tensor=pert_sign, params=params, labels=params['LABELS'], specific_class=params['TARGET_CLASS'])
    print("Prediction on the original sign:")
    test_on_tensor(model=model, tensor=orig_sign, params=params, labels=params['LABELS'])
    test_on_tensor(model=model, tensor=orig_sign, params=params, labels=params['LABELS'], specific_class=params['TARGET_CLASS'])