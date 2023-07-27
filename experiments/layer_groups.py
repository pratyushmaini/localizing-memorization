# This file will define the names of the layers that we will group together in our experiments
vit_groups = {0: ['to_patch_embedding.to_patch_tokens.1.weight',
                'to_patch_embedding.to_patch_tokens.1.bias',
                'to_patch_embedding.to_patch_tokens.2.weight',
                'to_patch_embedding.to_patch_tokens.2.bias',
                'transformer.layers.0.0.norm.weight',],
               1: ['transformer.layers.0.0.norm.weight',    
                    'transformer.layers.0.0.norm.bias',    
                    'transformer.layers.0.0.fn.temperature',
                    'transformer.layers.0.0.fn.to_qkv.weight',
                    'transformer.layers.0.0.fn.to_out.0.weight',
                    'transformer.layers.0.0.fn.to_out.0.bias',
                    'transformer.layers.0.1.norm.weight',    
                    'transformer.layers.0.1.norm.bias',       
                    'transformer.layers.0.1.fn.net.0.weight',
                    'transformer.layers.0.1.fn.net.0.bias',
                    'transformer.layers.0.1.fn.net.3.weight',
                    'transformer.layers.0.1.fn.net.3.bias', ],
                2: ['transformer.layers.1.0.norm.weight',    
                'transformer.layers.1.0.norm.bias',    
                'transformer.layers.1.0.fn.temperature',
                'transformer.layers.1.0.fn.to_qkv.weight',
                'transformer.layers.1.0.fn.to_out.0.weight',
                'transformer.layers.1.0.fn.to_out.0.bias',
                'transformer.layers.1.1.norm.weight',          
                'transformer.layers.1.1.norm.bias',       
                'transformer.layers.1.1.fn.net.0.weight',
                'transformer.layers.1.1.fn.net.0.bias',
                'transformer.layers.1.1.fn.net.3.weight',
                'transformer.layers.1.1.fn.net.3.bias',],
                3: ['transformer.layers.2.0.norm.weight',    
                'transformer.layers.2.0.norm.bias',    
                'transformer.layers.2.0.fn.temperature',
                'transformer.layers.2.0.fn.to_qkv.weight',
                'transformer.layers.2.0.fn.to_out.0.weight',
                'transformer.layers.2.0.fn.to_out.0.bias',
                'transformer.layers.2.1.norm.weight',
                'transformer.layers.2.1.norm.bias',
                'transformer.layers.2.1.fn.net.0.weight',
                'transformer.layers.2.1.fn.net.0.bias',
                'transformer.layers.2.1.fn.net.3.weight',
                'transformer.layers.2.1.fn.net.3.bias'],
                4:['transformer.layers.3.0.norm.weight',
                'transformer.layers.3.0.norm.bias',
                'transformer.layers.3.0.fn.temperature',
                'transformer.layers.3.0.fn.to_qkv.weight',
                'transformer.layers.3.0.fn.to_out.0.weight',
                'transformer.layers.3.0.fn.to_out.0.bias',
                'transformer.layers.3.1.norm.weight',
                'transformer.layers.3.1.norm.bias',
                'transformer.layers.3.1.fn.net.0.weight',
                'transformer.layers.3.1.fn.net.0.bias',
                'transformer.layers.3.1.fn.net.3.weight',
                'transformer.layers.3.1.fn.net.3.bias',],
                5:['transformer.layers.4.0.norm.weight',
                'transformer.layers.4.0.norm.bias',
                'transformer.layers.4.0.fn.temperature',
                'transformer.layers.4.0.fn.to_qkv.weight',
                'transformer.layers.4.0.fn.to_out.0.weight',
                'transformer.layers.4.0.fn.to_out.0.bias',
                'transformer.layers.4.1.norm.weight',
                'transformer.layers.4.1.norm.bias',
                'transformer.layers.4.1.fn.net.0.weight',
                'transformer.layers.4.1.fn.net.0.bias',
                'transformer.layers.4.1.fn.net.3.weight',
                'transformer.layers.4.1.fn.net.3.bias',],
                6: ['transformer.layers.5.0.norm.weight',
                'transformer.layers.5.0.norm.bias',
                'transformer.layers.5.0.fn.temperature',
                'transformer.layers.5.0.fn.to_qkv.weight',
                'transformer.layers.5.0.fn.to_out.0.weight',
                'transformer.layers.5.0.fn.to_out.0.bias',
                'transformer.layers.5.1.norm.weight',
                'transformer.layers.5.1.norm.bias',
                'transformer.layers.5.1.fn.net.0.weight',
                'transformer.layers.5.1.fn.net.0.bias',
                'transformer.layers.5.1.fn.net.3.weight',
                'transformer.layers.5.1.fn.net.3.bias',],
                7: ['mlp_head.0.weight',
                'mlp_head.0.bias',
                'mlp_head.1.weight',
                'mlp_head.1.bias']

            }

resnet50_groups = {
    0: ['conv1.weight',                           
        'bn1.weight',                        
        'bn1.bias',],
    1:[                        
    'layer1.0.conv1.weight',                 
    'layer1.0.bn1.weight',                 
    'layer1.0.bn1.bias',                     
    'layer1.0.conv2.weight',               
    'layer1.0.bn2.weight',               
    'layer1.0.bn2.bias',               
    'layer1.0.conv3.weight',                
    'layer1.0.bn3.weight',                    
    'layer1.0.bn3.bias',                           
    'layer1.0.downsample.0.weight',           
    'layer1.0.downsample.1.weight',      
    'layer1.0.downsample.1.bias',      
    'layer1.1.conv1.weight',                 
    'layer1.1.bn1.weight',                 
    'layer1.1.bn1.bias',                     
    'layer1.1.conv2.weight',               
    'layer1.1.bn2.weight',  
    'layer1.1.bn2.bias',    
    'layer1.1.conv3.weight',
    'layer1.1.bn3.weight',  
    'layer1.1.bn3.bias',                       
    'layer1.2.conv1.weight',              
    'layer1.2.bn1.weight',
    'layer1.2.bn1.bias',
    'layer1.2.conv2.weight',
    'layer1.2.bn2.weight',
    'layer1.2.bn2.bias',
    'layer1.2.conv3.weight',
    'layer1.2.bn3.weight',
    'layer1.2.bn3.bias',
    ],
    2:[
    'layer2.0.conv1.weight',
    'layer2.0.bn1.weight',
    'layer2.0.bn1.bias',
    'layer2.0.conv2.weight',
    'layer2.0.bn2.weight',
    'layer2.0.bn2.bias',
    'layer2.0.conv3.weight',
    'layer2.0.bn3.weight',
    'layer2.0.bn3.bias',
    'layer2.0.downsample.0.weight',
    'layer2.0.downsample.1.weight',
    'layer2.0.downsample.1.bias',
    'layer2.1.conv1.weight',
    'layer2.1.bn1.weight',                                 
    'layer2.1.bn1.bias',                                   
    'layer2.1.conv2.weight',                               
    'layer2.1.bn2.weight',                                 
    'layer2.1.bn2.bias',                                   
    'layer2.1.conv3.weight',                               
    'layer2.1.bn3.weight',                                 
    'layer2.1.bn3.bias',                                   
    'layer2.2.conv1.weight',                               
    'layer2.2.bn1.weight',                                 
    'layer2.2.bn1.bias',
    'layer2.2.conv2.weight',
    'layer2.2.bn2.weight',
    'layer2.2.bn2.bias',
    'layer2.2.conv3.weight',
    'layer2.2.bn3.weight',
    'layer2.2.bn3.bias',
    'layer2.3.conv1.weight',
    'layer2.3.bn1.weight',
    'layer2.3.bn1.bias',
    'layer2.3.conv2.weight',
    'layer2.3.bn2.weight',
    'layer2.3.bn2.bias',
    'layer2.3.conv3.weight',
    'layer2.3.bn3.weight',
    'layer2.3.bn3.bias',
    ],
    3:
    [
    'layer3.0.conv1.weight',
    'layer3.0.bn1.weight',
    'layer3.0.bn1.bias',
    'layer3.0.conv2.weight',
    'layer3.0.bn2.weight',
    'layer3.0.bn2.bias',
    'layer3.0.conv3.weight',
    'layer3.0.bn3.weight',
    'layer3.0.bn3.bias',
    'layer3.0.downsample.0.weight',
    'layer3.0.downsample.1.weight',
    'layer3.0.downsample.1.bias',
    'layer3.1.conv1.weight',
    'layer3.1.bn1.weight',
    'layer3.1.bn1.bias',
    'layer3.1.conv2.weight',
    'layer3.1.bn2.weight',
    'layer3.1.bn2.bias',
    'layer3.1.conv3.weight',
    'layer3.1.bn3.weight',
    'layer3.1.bn3.bias',
    'layer3.2.conv1.weight',
    'layer3.2.bn1.weight',
    'layer3.2.bn1.bias',
    'layer3.2.conv2.weight',
    'layer3.2.bn2.weight',
    'layer3.2.bn2.bias',
    'layer3.2.conv3.weight',
    'layer3.2.bn3.weight',
    'layer3.2.bn3.bias',
    'layer3.3.conv1.weight',
    'layer3.3.bn1.weight',
    'layer3.3.bn1.bias',
    'layer3.3.conv2.weight',
    'layer3.3.bn2.weight',
    'layer3.3.bn2.bias',
    'layer3.3.conv3.weight',
    'layer3.3.bn3.weight',
    'layer3.3.bn3.bias',
    'layer3.4.conv1.weight',
    'layer3.4.bn1.weight',
    'layer3.4.bn1.bias',
    'layer3.4.conv2.weight',
    'layer3.4.bn2.weight',
    'layer3.4.bn2.bias',
    'layer3.4.conv3.weight',
    'layer3.4.bn3.weight',
    'layer3.4.bn3.bias',
    'layer3.5.conv1.weight',
    'layer3.5.bn1.weight',
    'layer3.5.bn1.bias',
    'layer3.5.conv2.weight',
    'layer3.5.bn2.weight',
    'layer3.5.bn2.bias',
    'layer3.5.conv3.weight',
    'layer3.5.bn3.weight',
    'layer3.5.bn3.bias',
    ],
    4:[
    'layer4.0.conv1.weight',
    'layer4.0.bn1.weight',
    'layer4.0.bn1.bias',
    'layer4.0.conv2.weight',
    'layer4.0.bn2.weight',
    'layer4.0.bn2.bias',
    'layer4.0.conv3.weight',
    'layer4.0.bn3.weight',
    'layer4.0.bn3.bias',
    'layer4.0.downsample.0.weight',
    'layer4.0.downsample.1.weight',
    'layer4.0.downsample.1.bias',
    'layer4.1.conv1.weight',
    'layer4.1.bn1.weight',
    'layer4.1.bn1.bias',
    'layer4.1.conv2.weight',
    'layer4.1.bn2.weight',
    'layer4.1.bn2.bias',
    'layer4.1.conv3.weight',
    'layer4.1.bn3.weight',
    'layer4.1.bn3.bias',
    'layer4.2.conv1.weight',
    'layer4.2.bn1.weight',
    'layer4.2.bn1.bias',
    'layer4.2.conv2.weight',
    'layer4.2.bn2.weight',
    'layer4.2.bn2.bias',
    'layer4.2.conv3.weight',
    'layer4.2.bn3.weight',
    'layer4.2.bn3.bias',
    ],
    5:[
    'fc.weight',
    'fc.bias']
    }

resnet9_groups = {
    0:['0.0.weight',
 '0.1.weight',
 '0.1.bias',],
 1:['1.0.weight',
 '1.1.weight',
 '1.1.bias',],
 2:
 ['2.module.0.0.weight',
 '2.module.0.1.weight',
 '2.module.0.1.bias',],
 3:['2.module.1.0.weight',
 '2.module.1.1.weight',
 '2.module.1.1.bias',],
 4:['3.0.weight',
 '3.1.weight',
 '3.1.bias',],
 5:['5.module.0.0.weight',
 '5.module.0.1.weight',
 '5.module.0.1.bias',],
 6:['5.module.1.0.weight',
 '5.module.1.1.weight',
 '5.module.1.1.bias',],
 7:['6.0.weight',
 '6.1.weight',
 '6.1.bias',],
 8:['9.weight']
}

# calculate diff in weghts of each layer of two resnet9 models in a list
# diff_list = [(model_1_param - model_2_param).abs().sum().detach().item() for model_1_param, model_2_param in zip(model.parameters(), model_copy.parameters())]