# visualization of the distance estimation results
import os
from os.path import dirname as opd
import pandas as pd
from numpy.random import randint as np_randint
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt

# %% params
data_dir = './data/orig-data/test-data/chasing/'
csv_path = data_dir + 'annotations.csv'
img_dir = data_dir+'image/'
result_dir = './results/test/CL-CSEN/'
result_name = 'VGG19_mr_0.5_predictions'
save_dir = './results/test/CL-CSEN/' + result_name + '/'
save_flag = True
img_suf = '.png'
show_delay = 1000

#%% load data
df = pd.read_csv(csv_path)
resmat = sio.loadmat(result_dir+result_name+'.mat')
y_preds = resmat['y_preds'][0]
y_trues = resmat['y_trues'][0]

#%% visualziation
# create dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# show pred distance and true distance
# print('y_preds: ', y_preds, '\n', 'y_trues', y_trues)
# plt.figure()
plt.title('Pred. v.s. True distance')
plt.scatter(y_trues, y_preds, marker = 'o', s=40)
plt.xlabel("actual distance",fontsize=13)
plt.ylabel("predicted distance",fontsize=13)
if save_flag:
    plt.savefig(save_dir+'predVStrue.png')
plt.show()

# exit()
# visualze result estimation
last_img_name = ''
for idx, row in df.iterrows():
    
    img_name = row['filename'].replace('.txt', img_suf)
    
    if last_img_name==img_name:
        im = cv2.imread(save_dir + img_name)
    else:
        im = cv2.imread(img_dir + img_name)  # Load the image.
    
    # Object Location.
    x1 = int(row['xmin'])
    y1 = int(row['ymin'])
    x2 = int(row['xmax'])
    y2 = int(row['ymax'])
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    string = "(pred {:.2f}, true {:2f})".format(y_preds[idx], y_trues[idx])
    # text_color = np_randint(256,size=(1,3)).tolist()[0]
    cv2.putText(im, string, (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0,0,255], 1, cv2.LINE_AA)

    cv2.imshow("detections", im)
    if cv2.waitKey(show_delay) & 0xFF == ord('q'):
        break
    if save_flag:
        cv2.imwrite(save_dir+img_name, im)

    last_img_name = img_name
