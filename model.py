import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import numpy as np

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups = 1):
    result = models.Sequential()
    result.add(layers.Conv2D(out_channels,kernel_size = kernel_size,strides=stride,padding="same",name="conv",use_bias=False))
    result.add(layers.BatchNormalization(name="bn"))
    return result

class RepVGGBlock(layers.Layer):
    
    def __init__(self,in_channels,out_channels,kernel_size,stride = 1,padding=0,dilation=1,groups = 1,padding_mode='zeros',deploy=False):
        
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        
        assert kernel_size ==3 
        assert padding ==1
        padding_l1 = padding - kernel_size //2
        self.nonlinearity = layers.ReLU()
        
        if(deploy):
            self.rbr_reparam = layers.Conv2D(out_channels,kernel_size=kernel_size,strides = stride,padding = padding,dilation_rate = dilation,groups = groups,use_bias = True)
        else:
            
            self.rbr_identity = layers.BatchNormalization() if out_channels==in_channels and stride ==1 else None
            self.rbr_dense = conv_bn(in_channels = in_channels,out_channels = out_channels,kernel_size = kernel_size,stride = stride,padding = padding,groups = groups)
            self.rbr_1x1=conv_bn(in_channels=in_channels,out_channels = out_channels,kernel_size = 1,stride = stride,padding=padding_l1,groups = groups)
            print("RepVGG Block, identity= ",self.rbr_identity)
            
    def call(self,inputs):
        if(hasattr(self,"rbr_reparam")):
            return self.nonlinearity(self.rbr_reparam(inputs))
        
        if(self.rbr_identity is None):
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.rbr_dense(inputs)+self.rbr_1x1(inputs)+id_out)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3,bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1,bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid,biasid = self._fuse_bn_tensor(self.rbr_identity)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1)+kernelid,bias3x3+bias1x1+biasid
    def _pad_1x1_to_3x3_tensor(self,kernel1x1):
        if(kernel1x1 is None):
            return 0
        else:
            return tf.pad(kernel1x1,[[1,1],[1,1]],mode = 'CONSTANT',constant_values = 0)
    
    def _fuse_bn_tensor(self,branch):
        if(branch is None):
            return 0,0
        if(isinstance(branch,models.Sequential)):#模型conv_bn的
            kernel = branch.get_layer("conv").get_weights()[0] # 0 weights 1 bias
            bn_params = branch.get_layer("bn").get_weights()
            running_mean = bn_params[0]
            running_var = bn_params[1]
            gamma = bn_params[2]#weight
            beta = bn_params[3]#bias
        else:#直接过来的identity
            assert isinstance(branch,layers.BatchNormalization)
            if(not hasattr(self,'id_tensor')):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((3,3,self.in_channels,input_dim),dtype = np.float32)
                for i in range(self.in_channels):
                    kernel_value[1,1,i,i%input_dim] = 1
                self.id_tensor = tf.convert_to_tensor(kernel_value) #torch.from_numpy(kernel_value).to(branch.weight.device)#TODO
            kernel = self.id_tensor
            bn_params = branch.get_weights()
            running_mean = bn_params[0]
            running_var = bn_params[1]
            gamma = bn_params[2]#weight
            beta = bn_params[3]#bias
        std = tf.math.sqrt(running_var + 10e-8)
        t = (gamma / std).reshape(1,1,1,-1)
        
        return kernel * t ,beta - running_mean * gamma / std
    def repvgg_convert(self):
        kernel,bias = self.get_equivalent_kernel_bias()
        return kernel.numpy(),bias.numpy()

class RepVGG(layers.Layer):
    def __init__(self,num_blocks,num_classes=130,width_multiplier=None,override_groups_map = None,deploy = False):
        
        super(RepVGG,self).__init__()
        assert len(width_multiplier) == 4
        
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        
        assert 0 not in self.override_groups_map
        
        self.in_planes = min(64,int(64 * width_multiplier[0]))
        
        self.stage0 = RepVGGBlock(in_channels = 3,out_channels = self.in_planes,kernel_size =3,stride =2,padding = 1,deploy = self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64*width_multiplier[0]),num_blocks[0],stride = 2)
        self.stage2 = self._make_stage(int(128*width_multiplier[1]),num_blocks[1],stride = 2)
        
        self.stage3 = self._make_stage(int(256*width_multiplier[2]),num_blocks[2],stride = 2)
        self.stage4 = self._make_stage(int(512*width_multiplier[3]),num_blocks[3],stride = 2)
        self.gap = layers.GlobalAveragePooling2D()
        self.linear = layers.Dense(num_classes)
        
    def _make_stage(self,planes,num_blocks,stride):
        strides  =[stride] + [1]*(num_blocks -1)
        blocks = []
        model = models.Sequential()
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx,1)
            model.add(RepVGGBlock(in_channels=self.in_planes,out_channels = planes,kernel_size = 3,stride = stride,padding=1,groups = cur_groups,deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return model
    def call(self,x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        
        out = self.gap(out)
        #out = out.view(out.shape[1],-1)
        out = self.linear(out)
        
        return out
        
optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}                         

def create_RepVGG_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A1(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=130,width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A2(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)

def create_RepVGG_B0(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B1g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B2g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B2g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B3g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B3g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)


func_dict = {
'RepVGG-A0': create_RepVGG_A0,
'RepVGG-A1': create_RepVGG_A1,
'RepVGG-A2': create_RepVGG_A2,
'RepVGG-B0': create_RepVGG_B0,
'RepVGG-B1': create_RepVGG_B1,
'RepVGG-B1g2': create_RepVGG_B1g2,
'RepVGG-B1g4': create_RepVGG_B1g4,
'RepVGG-B2': create_RepVGG_B2,
'RepVGG-B2g2': create_RepVGG_B2g2,
'RepVGG-B2g4': create_RepVGG_B2g4,
'RepVGG-B3': create_RepVGG_B3,
'RepVGG-B3g2': create_RepVGG_B3g2,
'RepVGG-B3g4': create_RepVGG_B3g4,
}
def get_RepVGG_func_by_name(name):
    return func_dict[name]
