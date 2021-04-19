# 每一轮每个数据都能轮到的
import torch
import torch.utils.data as Data

BATCH_SIZE = 2

# 数据
x = torch.linspace(1 , 10 , 10)
y = torch.linspace(10 , 1 , 10)
print(x , y)
torch_dataset = Data.TensorDataset(x , y)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,     # 打乱顺序,但对应顺序还是没变的,加起来还是11
    )

for epoch in range(3):  # 全体数据训练三次
    for step , (batch_x , batch_y) in enumerate(loader):
        # training...
        print('Epoch: ' , epoch , '| Step: ' , step ,
              '| batch x: ' , batch_x.numpy() , 'batch y: ' , batch_y.numpy())
        
        
