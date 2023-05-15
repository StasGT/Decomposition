<H1>Low_cost == high_perfomance<br></H1>
Decomposition of convolutional networks, and 1 step to the side = 2 steps forward.<br>
In modern architectures of convolutional networks, the "BottleNeck" design is actively used, which performs the tasks of generalizing maps of features, decomposition, and, consequently, reducing the computational load. The effect is difficult to overestimate, it is astronomical.<br>
The 1x1 pixel kernel convolution indirectly participates in the CP-decomposition process, as in the image below:<br>
<img src="https://github.com/StasGT/Decomposition/blob/main/1.png" /><br>
To make it quite clear:<br>
<img src="https://github.com/StasGT/Decomposition/blob/main/2.png" /><br>
So, let's build the simplest ultralight convolutional network for classifying CIFAR100 images:<br>
self.conv 1 = nn.Conv2d(3, 96, 3, 1, 1) # 3 color maps of the image with a kernel=3x3 to 96 feature maps<br>
self.bn1 = nn.BatchNorm2d(96) # Normalize 96 feature maps<br>
self.conv2 = nn.Conv2d(96, 96, 3, 1, 1) # 96 feature maps with a kernel 3x3<br>
self.conv2n = nn.Conv2d(96, 32, 1) # 96 feature maps with a 1x1 kernel to 32 feature maps<br>
self.bn2n = nn.BatchNorm2d(32)<br>
self.conv3 = nn.Conv2d(32, 32, 5, stride=1, padding=2) # 32 map by kernel 5x5<br>
self.conn 4 = nn.Conv2d(32, 1, 1, 1, 0) # 32 map by kernel 1x1 to 1 map<br>
self.activation = nn.ReLU()<br>
self.fc1 = nn.Linear(32*32, 100) # Classify 1024 pixels into 100 classes<br>
<a href="https://github.com/StasGT/Decomposition/blob/main/CIFAR100_simpleNET.ipynb">CIFAR100_simpleNET</a><br>
Accuracy: 21% after 5 epochs / hmm, not very bad<br><br>
What if we add depth information between feature maps to the last convolutional layer...<br>
To do this, you can use Conv3d or transpose the tensor, process by Conv2d, transpose back and add to the stack of feature maps.<br><br>

xD1 = torch.transpose(x, 1, 2)   # We change axis Z and Y<br>
xD1 = self.conv3D2(xD1)          # Process<br>
xD1 = torch.transpose(xD1, 1, 2) # Change back<br><br>

x = torch.cat((xn, xD1, xD2), 1) # concatenation<br>
x = self.activation(self.bn7(x)) # normalization & activation<br>
<a href="https://github.com/StasGT/Decomposition/blob/main/CIFAR100_advanced_simple_NET.ipynb">CIFAR100_advanced_simple_NET</a><br>
So, this is much better. And this is with a negligibly small number of parameters - 282 925. This works the same as in the attention networks work key-value. I've trained this network to 45 percent accuracy and you can repeat.<br>
<br>Links:<br>
<a href="https://www.researchgate.net/publication/269935399_Speeding-up_Convolutional_Neural_Networks_Using_Fine-tuned_CP-Decomposition">Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition</a>
