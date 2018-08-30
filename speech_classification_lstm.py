import torch
import torch.nn as nn
import numpy as np
import glob
import matplotlib.pyplot as plt
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 13
hidden_size = 256
num_layers = 1
num_classes = 3
batch_size = 100
num_epochs = 6
learning_rate = 0.005
total_data = np.zeros(0,dtype='int32');
track = np.zeros(3,dtype='int32');
criterion = nn.CrossEntropyLoss()

def init():
    global total_data;
    data0 = glob.glob("trainning/0/*.wav")
    data1 = glob.glob("trainning/1/*.wav")
    data2 = glob.glob("trainning/2/*.wav")
    """
    data3 = glob.glob("trainning/3/*.wav")
    data4 = glob.glob("trainning/4/*.wav")
    data5 = glob.glob("trainning/5/*.wav")
    data6 = glob.glob("trainning/6/*.wav")
    data7 = glob.glob("trainning/7/*.wav")
    data8 = glob.glob("trainning/8/*.wav")
    data9 = glob.glob("trainning/9/*.wav")
    """
    for i in range(0, len(data0)):
        total_data = np.append(total_data, 0);
    for i in range(0, len(data1)):
        total_data = np.append(total_data, 1);
    for i in range(0, len(data2)):
        total_data = np.append(total_data, 2);
    """
    for i in range(0, len(data3)):
        a = np.append(a, 3);
    for i in range(0, len(data4)):
        a = np.append(a, 4);
    for i in range(0, len(data5)):
        a = np.append(a, 5);
    for i in range(0, len(data6)):
        a = np.append(a, 6);
    for i in range(0, len(data7)):
        a = np.append(a, 7);
    for i in range(0, len(data8)):
        a = np.append(a, 8);
    for i in range(0, len(data9)):
        a = np.append(a, 9);
    """
    np.random.shuffle(total_data);
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
def predict(input_line, label):
    print("Du doan:");
    with torch.no_grad():
        n_predictions = 3
        output = model(input_line)

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print(value," ",category_index);
        if (label == topi[0][0].item()):
            print(True);
            return True;
        else:
            print(False);
            return False;
init();
current_loss=0;
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
all_losses=[];
now = time.time();
model.load_state_dict(torch.load('model.ckpt'))
predict0=glob.glob("trainning/0_test/*.wav");
predict1=glob.glob("trainning/1_test/*.wav");
predict2=glob.glob("trainning/2_test/*.wav");
count=0;
for file in predict0:
    fs, audio = wav.read(file);
    inputs = mfcc(audio, samplerate=fs)
    inputs = torch.from_numpy(inputs).float();
    inputs = inputs.reshape(-1, inputs.size()[0], input_size).to(device)
    print(file);
    print(fs);
    if (predict(inputs,0) == True):
        count+=1;
for file in predict1:
    fs, audio = wav.read(file);
    inputs = mfcc(audio, samplerate=fs)
    inputs = torch.from_numpy(inputs).float()
    inputs = inputs.reshape(-1, inputs.size()[0], input_size).to(device)
    print(file);
    print(fs);
    if (predict(inputs,1)==True):
        count+=1;
for file in predict2:
    fs, audio = wav.read(file);
    inputs = mfcc(audio, samplerate=fs)
    inputs = torch.from_numpy(inputs).float();
    inputs = inputs.reshape(-1, inputs.size()[0], input_size).to(device)
    print(file);
    print(fs);
    if (predict(inputs, 2) == True):
        count += 1;
print("Predict:{}",(count/(len(predict1)+len(predict2)+len(predict0)))*100);
"""
a=0;
for i in range(100000000000):
    a=a+1;
    a=a-1;
"""
for i in range(num_epochs):
    plot_every = 0;
    current_loss = 0;
    print("Epoch ", i, " Start! ->");
    for j in range(len(total_data)):
        plot_every = plot_every + 1;
        stra = "trainning/" + str(total_data[j]) + "/" + str(track[total_data[j]]) + ".wav";
        track[total_data[j]] = track[total_data[j]] + 1;
        fs, audio = wav.read(stra);
        inputs = mfcc(audio, samplerate=fs)
        inputs = torch.from_numpy(inputs).float();
        inputs = inputs.reshape(-1, inputs.size()[0], input_size).to(device)
        labels = torch.tensor([int(total_data[j])], dtype=torch.long).to(device)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 1)
        optimizer.step()
        current_loss += loss
        if (plot_every>=batch_size):
            #all_losses.append(current_loss / plot_every)
            print("<> ",batch_size," pass ", current_loss / batch_size, " - Time ", time.time() - now, " !!!");
            now = time.time();
            plot_every = 0;
            current_loss = 0;
    track = np.zeros(3, dtype='int32');
    np.random.shuffle(total_data);
torch.save(model.state_dict(), 'model.ckpt')
#plt.figure()
#plt.plot(all_losses)
#plt.show()


#predict the data
