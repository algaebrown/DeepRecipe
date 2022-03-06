import torch
import torch.nn as nn
from torchvision import models
class ResNet_attention(nn.Module):
    ''' ResNet encoder'''

    def __init__(self, outdim=256, attn_channel = 1, 
                 ingd_embedsize = 50, n_ingd = 16, dropout_p = 0.2, n_lyr = 1):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        old_output_size = self.resnet.fc.in_features #[64, 1000]
        self.resnet_out_channel=2048
        self.resnet_outsize = 64
        
        # reduce resnet channel
        self.attn_channel = attn_channel
        self.reduce_channel = torch.nn.Conv2d(self.resnet_out_channel, attn_channel, kernel_size = (1,1)) # [64, 64, 8, 8]
        # make attn
        if n_lyr == 1:
            self.attn_fc = nn.Linear(self.attn_channel+ingd_embedsize, 1)
        elif n_lyr==2:
            self.attn_fc = nn.Sequential(nn.Linear(self.attn_channel+ingd_embedsize, 32), nn.ReLU(), nn.Linear(32,1))
        self.softmax = nn.Softmax2d()
        # reduce channel again
        self.squeeze_attention = torch.nn.Conv2d(n_ingd*attn_channel,1, kernel_size = (1,1)) 
        
        # predict embedding
        self.fc = nn.Linear(old_output_size, ingd_embedsize)
        
    def resnet_forward(self,images):
        with torch.no_grad(): # we don't want to ruin that pretained weights
            # I cannot remove avgpool with success :( even I remove avgpool it still return weirdly shaped outputs
            x1 = self.resnet.conv1(images)
            x1.shape # batchsize, 64 channels, size of img
            x2 = self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(x1))) 
            x3 = self.resnet.layer1(x2)
            x4 = self.resnet.layer2(x3)
            x5 = self.resnet.layer3(x4)
            features = self.resnet.layer4(x5) # 
        return features #[64, 2048, 8, 8]
        
    def forward(self, img, ingds_embedding, fine_tune = True):
        # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        
        batch_size = img.size()[0]
        n_input_ingd = ingds_embedding.size()[1]
        features = self.resnet_forward(img)  # [batchsize, 2048,8,8]
        
        # get attention mask based on image and the all input ingredients
        reduced = self.reduce_channel(features).permute(0,2,3,1).view(batch_size,self.resnet_outsize, self.attn_channel) # [batchsize, 64(pos),attn_channel]
        reduced = reduced.unsqueeze(1).repeat(1, n_input_ingd, 1, 1) # make 1 per gd [64, 4, 64, 1]
        ingds_repeat = ingds_embedding.unsqueeze(2).repeat(1,1,self.resnet_outsize, 1) #[batch_size, 50] # sum over all ingredients
        combined = torch.cat((reduced, ingds_repeat), dim = 3)
        
        attn_raw = self.attn_fc(combined).permute(0,1,3,2).view(batch_size, self.attn_channel*n_input_ingd, 8,8)
        attn = self.softmax(attn_raw) # batchsize, channel, 8*8
        mask = self.squeeze_attention(attn).repeat(1,self.resnet_out_channel,1,1)
        
        self.mask = mask # save mask for visualization
        
        # create masked output
        masked_features = mask*features
        
        # feed to fully connected
        pooled = self.resnet.avgpool(masked_features).view(batch_size, self.resnet_out_channel)
        output_ingd_embed = self.fc(pooled)

        return output_ingd_embed
