
import torch
import numpy as np
import cv2
from tqdm import tqdm
import State as State
from pixelwise_a3c import *
from FCN import *
from mini_batch_loader import MiniBatchLoader
import matplotlib.pyplot as plt
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

MOVE_RANGE = 9 #number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
EPISODE_LEN = 5
MAX_EPISODE = 5000
GAMMA = 0.95 
N_ACTIONS = 18
BATCH_SIZE = 32
DIS_LR = 3e-4
LR = 1e-3
IMG_SIZE = 70
SIGMA = 50
N_CHANNELS = 3
# TRAINING_DATA_PATH = "./train.txt"
# TESTING_DATA_PATH = "./train.txt"
TRAINING_DATA_PATH = "./CBSD68.txt"
TESTING_DATA_PATH = "./CBSD68.txt"
IMAGE_DIR_PATH = ".//"

def main():
    model = PPO(N_ACTIONS).to(device)
    # model.load_state_dict(torch.load("./torch_initweight/sig25_gray.pth"))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    i_index = 0

    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        IMG_SIZE)

    current_state = State.State((BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE), MOVE_RANGE)
    agent = PixelWiseA3C_InnerState(model, optimizer, BATCH_SIZE, EPISODE_LEN, GAMMA)
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    
    for n_epi in tqdm(range(0, MAX_EPISODE), ncols=70, initial=0):
        r = indices[i_index: i_index + BATCH_SIZE]
        # Load images
        raw_x = mini_batch_loader.load_training_data(r)
        # Create random noise for each image (mean=0, sigma=25)
        raw_n = np.random.normal(0, SIGMA, raw_x.shape).astype(raw_x.dtype) / 255
        current_state.reset(raw_x, raw_n)
        reward = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0

        if n_epi % 10 == 0:
            image = np.asanyarray(raw_x[10].reshape(IMG_SIZE,IMG_SIZE,N_CHANNELS) * 255, dtype=np.uint8)
            image = np.squeeze(image)
            cv2.imshow("rerr", image)
            # cv2.waitKey(1)

        for t in range(EPISODE_LEN):
            if n_epi % 10 == 0:
            #     # cv2.imwrite('./test_img/'+'ori%2d' % (t+c)+'.jpg', current_state.image[20].transpose(1, 2, 0) * 255)
                image = np.asanyarray(current_state.image[10].reshape(IMG_SIZE,IMG_SIZE,N_CHANNELS) * 255, dtype=np.uint8)
                image = np.squeeze(image)
                cv2.imshow("rerr", image)
                # cv2.waitKey(1)

            previous_image = np.clip(current_state.image.copy(), a_min=0., a_max=1.)
            action, inner_state, action_prob = agent.act_and_train(current_state.tensor, reward)

            if n_epi % 10 == 0:
                print(action[10])
                print(action_prob[10])
                # paint_amap(action[10])

            current_state.step(action, inner_state)
            reward = np.square(raw_x - previous_image) * 255 - np.square(raw_x - current_state.image) * 255
            sum_reward += np.mean(reward) * np.power(GAMMA, t)

        agent.stop_episode_and_train(current_state.tensor, reward, True)

        if i_index + BATCH_SIZE >= train_data_size:
            i_index = 0
            indices = np.random.permutation(train_data_size)
        else:
            i_index += BATCH_SIZE

        if i_index + 2 * BATCH_SIZE >= train_data_size:
            i_index = train_data_size - BATCH_SIZE

        print("train total reward {a}".format(a=sum_reward * 255))

    torch.save(model.state_dict(),f'./torch_pixel_model/pixel_sig{SIGMA}_color_individual_c.pth')
    
def paint_amap(acmap):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    # print(image)
    plt.imshow(image, vmin=1, vmax=9)
    plt.colorbar()
    plt.pause(1)
    # plt.show()
    plt.close()
if __name__ == '__main__':
    main()
