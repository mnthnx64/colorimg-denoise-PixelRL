
import torch
import numpy as np
import cv2
from FCN import PPO
from State import State
from torch.distributions import Categorical
import torch.optim as optim
from mini_batch_loader import MiniBatchLoader
from pixelwise_a3c import PixelWiseA3C_InnerState


def predict(img):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MOVE_RANGE = 3 #number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
    EPISODE_LEN = 5
    MAX_EPISODE = 2500
    GAMMA = 0.95 
    N_ACTIONS = 12
    LR = 1e-3
    IMG_SIZE = 70
    SIGMA = 25
    N_CHANNELS = 3
    IMAGE_DIR_PATH = ".//"
    # IMG_PATH = "./CBSD68-dataset/CBSD68/original_png/0004.png"

    model = PPO(N_ACTIONS).to(device)
    model.load_state_dict(torch.load('./torch_pixel_model/pixel_sig25_color.pth',  map_location='cpu'))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    agent = PixelWiseA3C_InnerState(model, optimizer, 1, EPISODE_LEN, GAMMA)
    model.eval()
    current_state = State((1, N_CHANNELS, IMG_SIZE, IMG_SIZE), move_range=MOVE_RANGE)
    mini_batch_loader = MiniBatchLoader(img, IMG_SIZE, True)
    sum_reward = 0
    
    # Load image
    raw_x = mini_batch_loader.load_validation_data()

    # Create random noise for image (mean=0, sigma=25)
    raw_n = np.random.normal(0, SIGMA, raw_x.shape).astype(raw_x.dtype)/255

    current_state.reset(raw_x,raw_n)
    reward = np.zeros(raw_x.shape, raw_x.dtype)

    for t in range(EPISODE_LEN):
        previous_image = current_state.image.copy()
        action, action_prob, inner_state = agent.act(current_state.tensor)
        current_state.step(action, inner_state)
        reward = np.square(raw_x - previous_image)*255 - np.square(raw_x - current_state.image)*255
        sum_reward += np.mean(reward)*np.power(GAMMA,t)

    original = np.asanyarray((raw_x*255), dtype=np.uint8)
    original = np.squeeze(original)

    noisy = np.clip(raw_x + raw_n, a_min=0., a_max=1.)
    noisy = np.asanyarray(noisy * 255, dtype=np.uint8)
    noisy = np.squeeze(noisy)
    noisy =  mini_batch_loader.stitch_image(noisy)

    corrected = np.asanyarray(current_state.image * 255, dtype=np.uint8)
    corrected = np.squeeze(corrected)
    corrected = mini_batch_loader.stitch_image(corrected)

    return {"total_reward": sum_reward, "prediction": corrected, "noisy": noisy}
