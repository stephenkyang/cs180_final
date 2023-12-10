import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image

model=models.vgg19(pretrained=True).features
device=torch.device( "cuda" if (torch.cuda.is_available()) else 'cpu')


def image_loader(path):
    image=Image.open(path)
    loader=transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
    image=loader(image).unsqueeze(0)
    return image.to(device,torch.float)

original_image=image_loader('venice.webp')
style_image=image_loader('mona_lisa.jpg')

generated_image=original_image.clone().requires_grad_(True)


class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.req_features= ['0','5','10','19','28'] 
        self.model=models.vgg19(pretrained=True).features[:29] 
    
   
    def forward(self,x):
        features=[]
        for layer_num,layer in enumerate(self.model):
            x=layer(x)
            if (str(layer_num) in self.req_features):
                features.append(x)
                
        return features
  
def calc_content_loss(gen_feat,orig_feat):
    content_l=torch.sum((gen_feat-orig_feat)**2)/2
    return content_l

def calc_style_loss(gen,style):
    batch_size,channel,height,width=gen.shape

    G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
    A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())

    style_l=torch.sum((G-A)**2)
    style_l = style_l/(4*(29**2)*(25**2))
    return style_l


def calculate_loss(gen_features, orig_feautes, style_featues):
    style_loss=content_loss=0
    for gen,cont,style in zip(gen_features,orig_feautes,style_featues):
        content_loss+=calc_content_loss(gen,cont)
        style_loss+=calc_style_loss(gen,style)
    
    total_loss=alpha*content_loss + beta*style_loss 
    return total_loss


model=VGG().to(device).eval() 

epoch= 400 #7000 before
lr= 0.005 #0.004 before
alpha= 10 # 8 before
beta= 10000 # 70 before

optimizer=optim.Adam([generated_image],lr=lr)


for e in range (epoch):
    gen_features=model(generated_image)
    orig_feautes=model(original_image)
    style_featues=model(style_image)
    
    total_loss=calculate_loss(gen_features, orig_feautes, style_featues)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if(e % 10 == 0):
        print(total_loss)
        
        save_image(generated_image,"images/mona_lisa(alpha10beta10000)/venice" + str(e//10) + ".png")