import os
import sys
import collections
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder

from models.missingTask.transformers_encoder.transformer import TransformerEncoder
import torch.nn.utils.rnn as rnn_utils


__all__ = ['CMFENet']

class CMFENet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.aligned = args.need_data_aligned
    
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)
        
        
       
        num_modality = 3
        self.fusion_method = args.fusion_method

        
        self.text_predictor = Predictor(768, 384, 64)
        self.audio_predictor = Predictor(64,32, 64)
        self.video_predictor = Predictor(64, 32, 64)

        ## ----------共享------
        self.text_projector2 = Projector(2432, 2432)
        self.audio_projector2 = Projector(2432, 2432)
        self.video_projector2 = Projector(2432, 2432)

        ## predictor
        self.text_predictor2 = Predictor(2432, 1216, 2432)
        self.audio_predictor2 = Predictor(2432, 1216, 2432)
        self.video_predictor2 = Predictor(2432, 1216, 2432)

        # low-level feature reconstruction
        self.recon_text = nn.Linear(args.d_model, args.feature_dims[0])
        self.recon_audio = nn.Linear(args.d_model, args.feature_dims[1])
        self.recon_video = nn.Linear(args.d_model, args.feature_dims[2])

        # final prediction module
        self.post_fusion_dropout = nn.Dropout(p=0.3)
        self.post_fusion_layer_1 = nn.Linear(7296, 1024)
        self.post_fusion_layer_2 = nn.Linear(1024, 256)
        self.post_fusion_layer_22 = nn.Linear(256, 64)
        self.post_fusion_layer_3 = nn.Linear(256, 1)

        self.player_a = nn.Linear(128, 32)
        self.player_b = nn.Linear(128, 32)
        self.player_c = nn.Linear(128, 32)
        

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.self_attentions_t_c = self.get_network(self_type='l')
        self.self_attentions_a_c = self.get_network(self_type='v')
        self.self_attentions_v_c = self.get_network(self_type='a')
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=1)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=1)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=1)

        

        self.recon_Linear1 = nn.Linear(64, 768, bias=False) 
        self.recon_Linear2 = nn.Linear(64, 33, bias=False) 
        self.recon_Linear3 = nn.Linear(64,709, bias=False) 
        
        self.pro_audio = nn.Linear(33, 32, bias=False) 
        self.pro_video = nn.Linear(709, 32, bias=False) 
        self.pro_text = nn.Linear(768, 32, bias=False) 
        self.proj1 = nn.Linear(64, 64)
        self.proj2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.p1 = nn.Linear(64, 32)
        self.p2 = nn.Linear(64, 32)
        self.p3 = nn.Linear(64, 32)
        # 定义自适应平均池化层
        self.adaptive_pool = nn.AdaptiveAvgPool1d(38)
    def get_network(self, self_type='l', layers=-1):
        self.num_heads =2
        self.layers =2
        self.attn_dropout = 0.3
        self.attn_dropout_a = 0.2
        self.attn_dropout_v = 0.0
        self.relu_dropout = 0.0
        self.embed_dropout = 0.2
        self.res_dropout = 0.0
        self.output_dropout = 0.5
        self.text_dropout = 0.5
        
        self.attn_mask = "true"
        
        if self_type in ['l']:
            embed_dim, attn_dropout = 32, self.attn_dropout
        elif self_type in ['a']:
            embed_dim, attn_dropout = 32, self.attn_dropout_a
        elif self_type in ['v']:
            embed_dim, attn_dropout = 32, self.attn_dropout_a
        
        elif self_type in [ 'la', 'va' ,'al', 'vl','lv', 'av']:
            embed_dim, attn_dropout = 32, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 64, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 64, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 64, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)


    def forward_once(self, text, text_lengths, audio, audio_lengths, video, video_lengths, missing):
        #SIMS
       #torch.Size([32, 3, 39]) torch.Size([32, 400, 33]) torch.Size([32, 55, 709])
        
        
        text = self.text_model(text)
        text_utt, text = text[:,0], text[:, 1:] # (B, 1, D), (B, T, D)orch.Size([32, 768]) torch.Size([32, 38, 768])
        text_for_recon = text.detach()#torch.Size([32, 38, 768])
        

        pro_text= self.pro_text(text)#torch.Size([[32, 38, 32])
        pro_audio= self.pro_audio(audio)#torch.Size([32, 400, 32]
        pro_video= self.pro_video(video)#torch.Size([32, 55, 32])
        
        text_view = pro_text.permute(1, 0, 2)#torch.Size([38, 32, 32])
        audio_view = pro_audio.permute(1, 0, 2)#torch.Size([400, 32, 32])
        video_view = pro_video.permute(1, 0, 2)#torch.Size([55, 32, 32])

        text_atten= text_view##torch.Size([38, 32, 32])
        audio_atten= audio_view##torch.Size([38, 32, 32])
        video_atten= video_view##torch.Size([38, 32, 32])
        
     
        
        TA_com = self.trans_l_with_a(text_atten, audio_atten, audio_atten)  # torch.Size([49, 32, 16])
        TV_com = self.trans_l_with_v(text_atten, video_atten, video_atten)  # Dimension (49,32, 16)
        T_com_cat = torch.cat([TA_com, TV_com], dim=2)#torch.Size(torch.Size([49, 32, 32])
        T_com_catsa = self.trans_l_mem(T_com_cat)
        # sigmoid 
        T_weight = torch.sigmoid(T_com_cat)            
        T_fin = T_com_catsa * T_weight                  
        T_fin = T_fin.permute(1, 0, 2)#torch.Size([32, 38, 64])
        


        AT_com = self.trans_a_with_l(audio_atten, text_atten, text_atten)  #torch.Size([375, 32, 16])
        AV_com = self.trans_a_with_v(audio_atten, video_atten, video_atten)  # Dimension ([375, 32, 16])
        A_com_cat = torch.cat([AT_com, AV_com], dim=2)#torch.Size(torch.Size([[375, 32,32])
        A_com_catsa = self.trans_a_mem(A_com_cat)
    #  sigmoid
        A_weight = torch.sigmoid(A_com_cat)          
        A_fin = A_com_catsa * A_weight                
        A_fin = A_fin.permute(1, 0, 2)#torch.Size([32, 400, 64])
        
      
        
        VT_com = self.trans_v_with_l(video_atten, text_atten, text_atten)  # torch.Size([500, 32, 16])
        VA_com = self.trans_v_with_a(video_atten, audio_atten, audio_atten)  # Dimension (49,32, 16)
        V_com_cat = torch.cat([VT_com, VA_com], dim=2)#torch.Size(torch.Size([49, 32, 32])
        V_com_catsa = self.trans_v_mem(V_com_cat)
     # ✅  sigmoid 
        V_weight = torch.sigmoid(V_com_cat)            # 
        V_fin = V_com_catsa * V_weight                   # 
        V_fin = V_fin.permute(1, 0, 2)#torch.Size([32, 55, 64])
        
        
        T_fusion = self.adaptive_pool(T_fin.transpose(1, 2)).transpose(1, 2)
        A_fusion = self.adaptive_pool(A_fin.transpose(1, 2)).transpose(1, 2)  # → [32, 38, 64]
        V_fusion = self.adaptive_pool(V_fin.transpose(1, 2)).transpose(1, 2)  # → [32, 38, 64]
       
       
        
        view_b_fused= T_fusion.reshape(T_fusion.size(0),-1)#torch.Size([32, 12544])
        view_a_fused= A_fusion.reshape(A_fusion.size(0),-1)#torch.Size([32, 12544])
        view_c_fused= V_fusion.reshape(V_fusion.size(0),-1)#torch.Size([32, 12544])
        fusion = torch.cat([view_b_fused,view_a_fused,view_c_fused], dim=-1)#torch.Size([32, 7350])
        
        fusion_h = self.post_fusion_dropout(fusion)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)
        
        
        #-------REC
        text_output = self.recon_Linear1(T_fin) #
        audio_output = self.recon_Linear2(A_fin) #
        video_output = self.recon_Linear3(V_fin) #

    


          #  SIM
        text_pre = view_b_fused
        audio_pre = view_a_fused
        video_pre =view_c_fused
        
        ## projector
        
        zz_text = self.text_projector2(text_pre)
        zz_audio = self.audio_projector2(audio_pre)
        zz_video = self.video_projector2(video_pre)

        ## predictor
        pp_text = self.text_predictor2(zz_text)
        pp_audio = self.audio_predictor2(zz_audio)
        pp_video = self.video_predictor2(zz_video)

            
            

        suffix = '_m' if missing else ''
        res = {
            f'pred{suffix}': output_fusion,  
            f'zz_text{suffix}': zz_text.detach(),
            f'pp_text{suffix}': pp_text,
            f'zz_audio{suffix}': zz_audio.detach(),
            f'pp_audio{suffix}': pp_audio,
            f'zz_video{suffix}': zz_video.detach(),
            f'pp_video{suffix}': pp_video,
           
        }

        # low-level feature reconstruction
        if missing:
            text_recon = text_output
            audio_recon = audio_output
            video_recon = video_output
            res.update(
                {
                    'text_recon': text_recon,
                    'audio_recon': audio_recon,
                    'video_recon': video_recon,

                    
                }
            )
        else:
            res.update({'text_for_recon': text_for_recon})
        

        return res

    def forward(self, text, audio, video):
        text, text_m = text
        audio, audio_m, audio_lengths = audio
        video, video_m, video_lengths = video
        
        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach() - 2 # -2 for CLS and SEP

        # complete view
        res = self.forward_once(text, text_lengths, audio, audio_lengths, video, video_lengths, missing=False)
        # incomplete view
        res_m = self.forward_once(text_m, text_lengths, audio_m, audio_lengths, video_m, video_lengths, missing=True)

        return {**res, **res_m}




class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(input_dim, output_dim),
                                 nn.BatchNorm1d(output_dim, affine=False))

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, input_dim, pred_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, pred_dim, bias=False),
                                 nn.BatchNorm1d(pred_dim),
                                 nn.ReLU(inplace=True),  # hidden layer
                                 nn.Linear(pred_dim, output_dim))  # output layer

    def forward(self, x):
        return self.net(x)


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask