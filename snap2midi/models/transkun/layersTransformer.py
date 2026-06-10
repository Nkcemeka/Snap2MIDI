import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utilities import checkpointByPass
import torch.utils.checkpoint

class RMSNorm(nn.Module):
    def __init__(self, eps = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(dim = -1, keepdim=True)
        return x* torch.rsqrt(var+ self.eps)


class TiedDropout(nn.Module):
    def __init__(self, dropoutProb, axis):
        super().__init__()
        self.dropout = nn.Dropout(dropoutProb)
        self.axis = axis
    
    def forward(self, x):

        if self.training:
            dropShape = list(x.shape)
            dropShape[self.axis] = 1
            mask = torch.ones(*dropShape, device = x.device)
        
            return self.dropout(mask)*x
        else:
            return x


class LearnableSpatialPositionEmbedding(nn.Module):
    def __init__(self, embedSize, coordDim, gamma = 10.0, dropoutProb = 0.0):
        super().__init__()

        self.gamma = gamma
        self.proj = nn.Linear(coordDim, embedSize)

        self.mlp = nn.Sequential(
                nn.Linear(embedSize, 4*embedSize),
                nn.GELU(),
                nn.Dropout(dropoutProb),
                nn.Linear(4*embedSize, embedSize))


        self.dropout = nn.Dropout(dropoutProb)
        self._reset_parameters()


    def _reset_parameters(self):
        nn.init.normal_(self.proj.weight, std = 1/self.gamma)
        nn.init.uniform_(self.proj.bias, a = -math.pi, b = math.pi)

    """
    arguments:
        indices [nBatch, nDimCoordinates]
    """
    def forward(self, *coords):
        device = self.proj.weight.device
        coords = torch.meshgrid(coords, indexing="ij")
        coord = torch.stack(coords, dim = -1)

        phi = self.proj(coord.float())

        z = torch.cos(phi)/ math.sqrt(phi.shape[-1]/2)
        z = self.mlp(z)

        return z

    def forwardWithCoordVec(self, coord):
        device = self.proj.weight.device

        phi = self.proj(coord.float())

        z = torch.cos(phi)/ math.sqrt(phi.shape[-1]/2)
        z = self.mlp(z)

        return z

class ResBlock(nn.Module):
    def __init__(self, module, size, prenorm = True, dropoutProb =0.0):
        super().__init__()
        self.module = module
        self.norm = RMSNorm()

        # LayerScale
        self.scale = nn.Parameter(torch.ones(size)*1e-2)
        self.dropout = nn.Dropout(dropoutProb)

    def forward(self, x, *args):
        return x + self.dropout(self.module(self.norm(x), *args))*self.scale

class SelfAttnWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        shape = x.shape
        x = x.flatten(0,1)
        result, _ = self.module(x,x,x)

        result = result.unflatten(0, shape[:2])
        return result


"""
The customized MHA layer using the approximated attention
adapted from the pytorch implementation
"""
class MultiHeadAttentionKernel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout = 0., k_dim = None, v_dim = None, fourierSize = 32, kernel = "fourier", hiddenFactor = 1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kernel = kernel


        hiddenSize = math.ceil(hiddenFactor*embed_dim)

        # make sure hiddenSize to be divisible by num_heads
        self.head_dim = int(math.ceil(hiddenSize/num_heads))
        hiddenSize = self.head_dim*num_heads

        if k_dim is None:
            k_dim = embed_dim

        if v_dim is None:
            v_dim = embed_dim


        self.fourierSize = fourierSize


        self.q_proj_weight = nn.Parameter(torch.empty(( embed_dim, hiddenSize)))
        self.k_proj_weight = nn.Parameter(torch.empty(( k_dim, hiddenSize)))
        self.v_proj_weight = nn.Parameter(torch.empty(( v_dim, hiddenSize)))
        self.out_proj = nn.Linear(hiddenSize, embed_dim)

        if kernel is not None:
            self.gamma = nn.Parameter(torch.tensor(1.0))
            self.norm = RMSNorm()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)

    def forward(self, query, key = None, value = None):
        if key == None:
            key = query

        if value == None:
            value = key

        q = query@self.q_proj_weight
        k = key@self.k_proj_weight
        v = value@self.v_proj_weight

        # split into heads
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = q.transpose(-2,-3)
        k = k.unflatten(-1, (self.num_heads, self.head_dim))
        k = k.transpose(-2,-3)
        v = v.unflatten(-1, (self.num_heads, self.head_dim))
        v = v.transpose(-2,-3)

        if self.kernel is not None:
            raise NotImplementedError
        else:
            fetched = F.scaled_dot_product_attention(q,k,v)

        fetched = fetched.transpose(-2,-3).flatten(-2,-1)

        result = self.out_proj(fetched)

        return result


class BasicBlock(nn.Module):
    def __init__(self,
            inputSize,
            num_heads,
            fourierSize,
            hiddenFactor = 2,
            hiddenFactorAttn = 1,
            approxKernels = [None,
                None,
                None, None],
            enabled = ["F", "T", "All0", "0All"],
            dropoutProb = 0.0):
        super().__init__()

        fnnHiddenSize = int(math.ceil(inputSize*hiddenFactor))

        self.enabled = enabled

        if "F" in enabled:
            self.mhaBlockF = ResBlock(
                    MultiHeadAttentionKernel(
                        inputSize,
                        num_heads=num_heads,
                        fourierSize = fourierSize,
                        kernel = approxKernels[0],
                        hiddenFactor = hiddenFactorAttn),

                    size = inputSize,
                    dropoutProb = dropoutProb
                    )

            self.fnnBlockF  = ResBlock(
                    nn.Sequential(
                        nn.Linear(inputSize, fnnHiddenSize),
                        nn.GELU(),
                        nn.Dropout(dropoutProb),
                        nn.Linear(fnnHiddenSize, inputSize),
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )

        if "T" in enabled:
            self.mhaBlockT = ResBlock(
                    MultiHeadAttentionKernel(
                        inputSize,
                        num_heads=num_heads,
                        fourierSize = fourierSize,
                        kernel = approxKernels[1],
                        hiddenFactor = hiddenFactorAttn
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )
            self.fnnBlockT  = ResBlock(
                    nn.Sequential(
                        nn.Linear(inputSize, fnnHiddenSize),
                        nn.GELU(),
                        nn.Dropout(dropoutProb),
                        nn.Linear(fnnHiddenSize, inputSize),
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )
        if "All0" in enabled or "0All" in enabled :
            self.mhaBlockAll0 = ResBlock(
                    MultiHeadAttentionKernel(
                        inputSize,
                        num_heads=num_heads,
                        fourierSize = fourierSize,
                        kernel = approxKernels[2],
                        hiddenFactor = hiddenFactorAttn
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )
            self.fnnBlockAll0  = ResBlock(
                    nn.Sequential(
                        nn.Linear(inputSize, fnnHiddenSize),
                        nn.GELU(),
                        nn.Dropout(dropoutProb),
                        nn.Linear(fnnHiddenSize, inputSize),
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )

        if "FT" in enabled:
            self.mhaBlockFT = ResBlock(
                    MultiHeadAttentionKernel(
                        inputSize,
                        num_heads=num_heads,
                        fourierSize = 64,
                        kernel = "positive",
                        hiddenFactor = hiddenFactorAttn
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )

            self.fnnBlockFT = ResBlock(
                    nn.Sequential(
                        nn.Linear(inputSize, fnnHiddenSize),
                        nn.GELU(),
                        nn.Dropout(dropoutProb),
                        nn.Linear(fnnHiddenSize, inputSize),
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )



    def forward(self, x, mem = None, crossAttn = False):
        # x: [N, T, F, D]
        nT = x.shape[-3]
        nF = x.shape[-2]

        inShape = x.shape

        crossAttn = True 

        if mem is None:
            mem = x
            crossAttn = False

        h = x

        if "F" in self.enabled:
            # print("F")
            h = self.mhaBlockF(h, mem)
            h = self.fnnBlockF(h)

        # change to [N, F, T, D]
        h = h.transpose(-3, -2)
        mem = mem.transpose(-3, -2)

        if "T" in self.enabled:
            # print("T")
            # all attends to the aggregated track
            # if crossAttn:
                # h = self.mhaBlockT(h, mem[..., 0:1, :, :])
            # else:
            h = self.mhaBlockT(h, mem)
            h = self.fnnBlockT(h)

        if "All0" in self.enabled or "0All" in self.enabled:
            h0, h1 = h.split([1, h.shape[-3]-1], dim = -3)

            if "All0" in self.enabled:
                h1 = self.mhaBlockAll0(h1, mem[..., 0:1, :,:])

            if "0All" in self.enabled:
                # h0 = h[..., 0, :, :]
                h0 = self.mhaBlockAll0(h0, mem.flatten(-3,-2).unsqueeze(-3))
                # print(h.shape)

            h = torch.cat([h0, h1], dim = -3)
            h = self.fnnBlockAll0(h)
            
        # change to [N, F*T, D]
        if "FT" in self.enabled:
            h = h.flatten(-3,-2)
            mem = mem.flatten(-3, -2)

            h = self.mhaBlockFT(h, mem)
            h = self.fnnBlockFT(h)
            h = h.unflatten(-2, (nF, nT))

        # change back to the orignal shape
        h = h.transpose(-3, -2)

        outShape = h.shape
        assert inShape == outShape
        return h

"""
What if we even simplify the scoring module?
"""
class ScaledInnerProductIntervalScorer(nn.Module):
    def __init__(self, size, expansionFactor = 1, dropoutProb = 0.0, withScoreEps = False, lengthScaling="linear"):
        super().__init__()

        self.size = size

        if not withScoreEps:
            self.map = nn.Sequential(
                    nn.Linear(size, 2*size*expansionFactor+1), # only inner product plus diagonal
                    )
        else:
            self.map = nn.Sequential(
                    nn.Linear(size, 2*size*expansionFactor+1 + 1), 
                    )

        
        self.dropout = nn.Dropout(dropoutProb)

        self.expansionFactor = expansionFactor

        self.lengthScaling = lengthScaling

    def forward(self, ctx):


        q, k, diag = (self.map(ctx)).split([self.size*self.expansionFactor,
            self.size*self.expansionFactor, 1], dim = -1)

        # print(q.std(), k.std(), b.std())
        q = q/math.sqrt(q.shape[-1])

        # part1 innerproduct
        S = torch.einsum("iped, ipbd-> ipeb", q, k)
        # diagS = S.diagonal(dim1= -2, dim2=-1)

        tmpIdx_e = torch.arange(S.shape[-2], device = S.device)
        tmpIdx_b = torch.arange(S.shape[-1], device = S.device)
        len_eb = (tmpIdx_e.unsqueeze(-1)- tmpIdx_b.unsqueeze(0)).abs()

        if self.lengthScaling == "linear":
            S = S*(len_eb)
        elif self.lengthScaling == "sqrt":
            S = S*(len_eb).float().sqrt()
        elif self.lengthScaling == "none":
            pass
        else:
            raise Exception("Unrecognized lengthScaling")



        diagM= torch.diag_embed(diag.squeeze(-1))

        S = S + diagM

        # dummy eps score for testing
        b = diag*0.0
        b = b[...,1:, 0]

        S = S.permute(2,3, 0, 1).contiguous()
        b = b.permute(2,0,1).contiguous()
        return S, b


class Backbone(nn.Module):
    def __init__(self,
            inputSize,
            baseSize,
            posEmbedInitGamma,
            nHead,
            fourierSize = 16,
            hiddenFactor = 2,
            hiddenFactorAttn = 1,
            expansionFactor = 1,
            dropoutProb = 0.0,
            nLayers= 4,
            enabledAttn = ["F", "T"],
            useGradientCheckpoint = True,
            downsampleF = True,
            upsampleProjOnly = True,
            ) :
        super().__init__()

        self.posEmbedBuilder = LearnableSpatialPositionEmbedding(
                baseSize,
                coordDim = 1,
                gamma = posEmbedInitGamma,
                dropoutProb= dropoutProb)




        self.inputConv = nn.Conv2d(inputSize, baseSize, kernel_size = 3, padding =1)
        self.dropoutTied = TiedDropout(dropoutProb, axis = -3)

        # temporal patch of size 8

        if not downsampleF:
            # path size 8x1
            self.downConv = nn.Sequential(
                    nn.ConstantPad2d( (0,0, 4, 3), value = 0.0),
                    nn.Conv2d(baseSize, baseSize*2, 3, padding = 1, stride = (2,1)),
                    nn.GroupNorm(4, baseSize*2),
                    nn.GELU(),
                    nn.Dropout2d(dropoutProb),

                    nn.Conv2d(baseSize*2, baseSize*4, 3,  padding = 1, stride = (2,1)),
                    nn.GroupNorm(4, baseSize*4),
                    nn.GELU(),
                    nn.Dropout2d(dropoutProb),

                    nn.Conv2d(baseSize*4, baseSize*4, 3,  padding = 1, stride = (2,1)),
                    nn.GroupNorm(4, baseSize*4),
                    nn.GELU(),
                    nn.Dropout2d(dropoutProb),
                    nn.Conv2d(baseSize*4, baseSize*4, 3,  padding = 1),
                    nn.GroupNorm(4, baseSize*4),
                    )
        else:
            # this time the patch size is 8x 4
            self.downConv = nn.Sequential(
                    nn.ConstantPad2d( (2, 1, 4, 3), value = 0.0),
                    nn.Conv2d(baseSize, baseSize*2, 3, padding = 1, stride = (2,1)),
                    nn.GroupNorm(4, baseSize*2),
                    nn.GELU(),
                    nn.Dropout2d(dropoutProb),

                    nn.Conv2d(baseSize*2, baseSize*4, 3,  padding = 1, stride = (2,2)),
                    nn.GroupNorm(4, baseSize*4),
                    nn.GELU(),
                    nn.Dropout2d(dropoutProb),

                    nn.Conv2d(baseSize*4, baseSize*4, 3,  padding = 1, stride = (2,2)),
                    nn.GroupNorm(4, baseSize*4),
                    nn.GELU(),
                    nn.Dropout2d(dropoutProb),
                    nn.Conv2d(baseSize*4, baseSize*4, 3,  padding = 1),
                    nn.GroupNorm(4, baseSize*4),
                    )


        self.upConv1dSkip =  nn.ConvTranspose1d(baseSize*4, baseSize*expansionFactor, 8, stride = 8)
        if not upsampleProjOnly:
            self.upConv1d = nn.Sequential(
                    nn.ConvTranspose1d(baseSize*4, baseSize*4, 2, stride = 2),
                    nn.Conv1d(baseSize*4, baseSize*4, 3, padding = 1),
                    nn.GroupNorm(4,  baseSize*4),
                    nn.GELU(),
                    nn.ConvTranspose1d(baseSize*4, baseSize*2, 2, stride = 2),
                    nn.Conv1d(baseSize*2, baseSize*2, 3, padding = 1),
                    nn.GroupNorm(4,  baseSize*2),
                    nn.GELU(),
                    nn.ConvTranspose1d(baseSize*2, baseSize, 2, stride = 2),
                    nn.Conv1d(baseSize, baseSize, 3, padding = 1),
                    )
        self.upsampleProjOnly = upsampleProjOnly

        self.posEmbedBuilderAttnTF = LearnableSpatialPositionEmbedding(
                baseSize*4,
                coordDim = 2,
                gamma = posEmbedInitGamma,
                dropoutProb=dropoutProb)


        self.posEmbedBuilderAttnTE = LearnableSpatialPositionEmbedding(
                baseSize*4,
                coordDim = 2,
                gamma = posEmbedInitGamma, dropoutProb = dropoutProb)
          
        encoderLayers = [BasicBlock( inputSize = baseSize*4,
            num_heads = nHead,
            fourierSize = fourierSize,
            dropoutProb = dropoutProb,
            hiddenFactor = hiddenFactor,
            hiddenFactorAttn = hiddenFactorAttn,
            enabled = enabledAttn
            ) for i in range(nLayers)]

        self.encoderLayers = nn.ModuleList(encoderLayers)

        self.normEncoder = nn.Identity()

        self.useGradientCheckpoint = useGradientCheckpoint

        self.dropout = nn.Dropout(dropoutProb)

    def forward(self, x, outputIndices):
        if self.useGradientCheckpoint or self.training:
            checkpoint = torch.utils.checkpoint.checkpoint
        else:
            checkpoint = checkpointByPass
        # x: [N, T, F, D]

        # change to
        # x: [N, D, T, F]
        x = x.permute(0, 3,1,2)


        nT = x.shape[-2]
        nF = x.shape[-1]

        # append the first position embedding

        coord_F = torch.arange(x.shape[-1], device = x.device).float()

        posEmbedInputConv = self.posEmbedBuilder(coord_F)
        posEmbedInputConv = posEmbedInputConv.transpose(-1,-2).unsqueeze(-2)


        # downsample along the the time

        # downsample 4 times for reducing the computation cost
        h = self.inputConv(x)+posEmbedInputConv
        h = self.downConv(h)


        # change to [N, T, F, D] shape
        h = h.permute(0, 2, 3, 1)

        # append 1 time and 1 frequency aggregation track
        h = F.pad(h, (0,0,1,0, 1, 0))

        ################ transformer encoders
        coord_F = torch.arange(h.shape[-2], device = x.device).float()

        coord_T = torch.arange(h.shape[-3], device = x.device).float()
        outputIndices =  outputIndices.float()

        posEmbed = self.posEmbedBuilderAttnTF(coord_T, coord_F)

        posEmbedTgt = self.posEmbedBuilderAttnTE(coord_T,  outputIndices)

        posEmbedTgt = posEmbedTgt.unsqueeze(0).repeat(h.shape[0], 1,1,1)

        h = h + posEmbed
        hTarget = posEmbedTgt
        hAll = torch.cat( [h, hTarget], dim = -2)


        for l in self.encoderLayers:
            hAll = checkpoint(l, hAll, use_reentrant=False)

        h, hTarget = hAll.split([h.shape[-2], hTarget.shape[-2]], dim = -2)

        # print(hTarget.std(), hTarget.mean())

        # remove the t=0 pooling track
        hTarget = hTarget[..., 1:, :, :]

        # do 1d upsampling
        hTarget = hTarget.permute(0, 2, 3,1)

        # now h:[N, P, D,T ]

        hTarget = hTarget.flatten(0,1)
        # now h:[N*P, D,T ]
        if not self.upsampleProjOnly:
            hTarget = self.upConv1d(hTarget)+ self.upConv1dSkip(hTarget)
        else:
            # it seems that this linear projection is good enough
            hTarget = self.upConv1dSkip(hTarget)

        hTarget = hTarget.unflatten(0, (x.shape[0], len(outputIndices)))

        hTarget = hTarget[..., :nT]

        hTarget = hTarget.permute(0, 1,3,2)

        # the outputShape: [N, P, T, D]
        return hTarget
