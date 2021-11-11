
import torch
import numpy as np
import cv2
from FCN import PPO
from State import State
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import torch.optim as optim
from mini_batch_loader import MiniBatchLoader
from pixelwise_a3c import PixelWiseA3C_InnerState
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MOVE_RANGE = 3
EPISODE_LEN = 5
MAX_EPISODE = 2500
GAMMA = 0.95 
N_ACTIONS = 12
LR = 1e-3
IMG_SIZE = 63
SIGMA = 25
CHANNELS = 3
IMAGE_DIR_PATH = ".//"
IMG_PATH = "./CBSD68/original_png/0001.png"

model = PPO(N_ACTIONS).to(device)
model.load_state_dict(torch.load('./torch_pixel_model/pixel_sig25_color.pth'))
optimizer = optim.Adam(model.parameters(), lr=LR)
agent = PixelWiseA3C_InnerState(model, optimizer, 1, EPISODE_LEN, GAMMA)


def tst(model, agent):
    model.eval()
    current_state = State((1, CHANNELS, IMG_SIZE, IMG_SIZE), move_range=MOVE_RANGE)
    mini_batch_loader = MiniBatchLoader(IMG_PATH, IMG_PATH, IMAGE_DIR_PATH, IMG_SIZE, True)
    sum_reward = 0
    
    # Load image
    raw_x = mini_batch_loader.load_validation_data()
    # raw_x = cv2.imread(IMG_PATH,1).astype(np.float32)
    # raw_x = cv2.cvtColor(raw_x, cv2.COLOR_RGB2GRAY)
    # raw_x = (raw_x/255).astype(np.float32).reshape(3,63,63)

    # Create random noise for image (mean=0, gamma=25)
    raw_n = np.random.normal(0, SIGMA, raw_x.shape).astype(raw_x.dtype)/255

    current_state.reset(raw_x,raw_n)
    reward = np.zeros(raw_x.shape, raw_x.dtype)

    for t in range(EPISODE_LEN):
        previous_image = current_state.image.copy()
        action, action_prob, inner_state = agent.act(current_state.tensor)
        current_state.step(action, inner_state)
        reward = np.square(raw_x - previous_image)*255 - np.square(raw_x - current_state.image)*255
        sum_reward += np.mean(reward)*np.power(GAMMA,t)

    original = np.asanyarray((raw_x[0]*255+0.5).reshape(63,63,3), dtype=np.uint8)
    original = np.squeeze(original)
    cv2.imshow("Original", original)
    # cv2.waitKey(0)

    noisy = np.clip(raw_x + raw_n, a_min=0., a_max=1.)
    noisy = np.asanyarray(noisy[0].reshape(63,63,3) * 255, dtype=np.uint8)
    noisy = np.squeeze(noisy)
    cv2.imshow("Noisy", noisy)
    # cv2.waitKey(0)

    # s = np.clip(raw_x + raw_n, a_max=1., a_min=0.)
    # ht = np.zeros([s.shape[0], 64, s.shape[2], s.shape[3]], dtype=np.float32)
    # st = np.concatenate([s, ht], axis=1)
    # action_map, action_map_prob, ht_ = select_action(torch.FloatTensor(st).to(device), test=True)  # 1, 3, 63, 63
    # step_test.set(st)
    # paint_amap(action_map_prob[0])
    # print(action_map[0])
    # print(action_map_prob[0])

    corrected = np.asanyarray(current_state.image[0].reshape(63,63,3) * 255, dtype=np.uint8)
    corrected = np.squeeze(corrected)
    cv2.imshow("corrected", corrected)
    cv2.waitKey(0)

    print("test total reward {a}".format(a=sum_reward))

# def select_action(state, test=False):
#     with torch.no_grad():
#         pout, val_, ht_ = model(state)
#     pout = torch.clamp(pout, min=0, max=1)
#     p_trans = pout.permute([0, 2, 3, 1])
#     dist = Categorical(p_trans)
#     if test:
#         _, action = torch.max(pout, dim=1)
#     else:
#         action = dist.sample().detach()  # action

#     action_prob = pout.gather(1, action.unsqueeze(1))
#     return action.unsqueeze(1).detach().cpu(), action_prob.detach().cpu(), ht_.detach().cpu()

def paint_amap(acmap):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image,  vmin=1, vmax=9)
    plt.colorbar()
    plt.show()

tst(model, agent)













