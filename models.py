#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:10:01 2020

@author: allan
"""

import layers
import torch
import torch.nn as nn
import sys
from transformers import AlbertModel
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


size_map = {'albert-base-v2':768, 'albert-large-v2':1024, 'albert-xlarge-v2':2048, 'albert-xxlarge-v2':4096}
    
class AlbertLinear_highway(nn.Module):
    def __init__(self, model_name):
        super(AlbertLinear_highway, self).__init__()
        self.albert = AlbertModel.from_pretrained(model_name)
        input_dim = size_map[model_name]
        self.qa_outputs = nn.Linear(input_dim , 2)
        self.enc = layers.HighwayEncoder(3, input_dim)
        
    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        outputs = self.albert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    )

        sequence_output = outputs[0]
        sequence_output = self.enc(sequence_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs
    
class LSTM_highway(nn.Module):
    def __init__(self, model_name, hidden_size, drop_prob=0.):
        super(LSTM_highway, self).__init__()
        self.albert = AlbertModel.from_pretrained(model_name)
        input_size = size_map[model_name]
        self.enc = layers.RNNEncoder(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=2,
                                drop_prob=drop_prob)
        self.dec = layers.RNNEncoder(input_size=2 * hidden_size,
                                hidden_size=hidden_size,
                                num_layers=2,
                                drop_prob=drop_prob)
        self.highway = layers.HighwayEncoder(2, 2 * hidden_size)
        self.qa_outputs = nn.Linear(2 * hidden_size, 2)
        
    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        c_mask = (attention_mask == 1) * (token_type_ids == 1)
        for i in range(c_mask.shape[0]):
            c_mask[i, 0] = 1
        length = attention_mask.sum(-1)
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        enc = self.enc(sequence_output, length)
        proj = self.highway(enc)
        dec = self.dec(proj, length)
        logits = self.qa_outputs(dec)
        start_logits, end_logits = logits.split(1, dim=-1)
        mask = c_mask.type(torch.float32)
        start_logits = mask * start_logits.squeeze(-1) + (1 - mask) * -1e30
        end_logits = mask * end_logits.squeeze(-1) + (1 - mask) * -1e30

        outputs = (start_logits, end_logits,)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs

class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.
    Layers:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple BiDAF layer to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.hidden_size = hidden_size * 2 # adding the char embedding, double the hidden_size. 
        
        self.emb = layers.Embedding(model_name='albert-base-v2',
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        #input_size=self.hidden_size+2 is due to we add two extra features (avg_attention) to both char embedding
        #and word embedding to boost the performance. The avg_attention is use the attention mechanism to learn 
        #a weighted average among the vectors by the model itself.
        self.enc = layers.RNNEncoder(input_size=self.hidden_size+2,
                                     hidden_size=self.hidden_size,
#                                      num_layers=2, # The number of layer can be changed, but less or no improvement.
                                     num_layers = 1,
                                     drop_prob=drop_prob)
        
        self.att = layers.BiDAFAttention(hidden_size=2*self.hidden_size,
                                         drop_prob=drop_prob)
        
        #Add extra layer of self-attention based on the paper 'Simple and Effective Multi-Paragraph Reading Comprehension'
        #URL: https://arxiv.org/pdf/1710.10723.pdf 
        self.self_att = layers.SelfAtt(hidden_size=2*self.hidden_size,
                                      drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8*self.hidden_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

#         self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
       
        self.out = layers.BiDAFOutput(hidden_size=self.hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, char_ids, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        c_mask = (attention_mask == 1) * (token_type_ids == 1)
        q_mask = (attention_mask == 1) * (token_type_ids == 0)
        for i in range(c_mask.shape[0]):
            c_mask[i, 0] = 1
        
        c_len, q_len = attention_mask.sum(-1), q_mask.sum(-1)

        
        emb = self.emb(c=char_ids,
                       input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids
                       )
        
        q_emb = emb * q_mask[:,:,None].float()
        c_emb = emb * c_mask[:,:,None].float()
        # c_embs, c_masks = [], []
        # for i in range(input_ids.shape[0]):
        #     c_embs.append(torch.cat((c_emb[i, q_len[i]:], c_emb[i, :q_len[i]]), dim=0))
        #     c_masks.append(torch.cat((a_c_mask[i, q_len[i]:], a_c_mask[i, :q_len[i]]), dim=0))
        # c_emb = torch.stack(c_embs, dim=0)
        # c_mask = torch.stack(c_masks, dim=0)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        att = self.self_att(att, c_mask)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        start_logits, end_logits = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)
        # start_logits = start_logits.squeeze(-1)
        # end_logits = end_logits.squeeze(-1)
        outputs = (start_logits, end_logits,) 
        
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs
    
class BiDAF2(nn.Module):
    """Baseline BiDAF model for SQuAD.
    Layers:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple BiDAF layer to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, model_name, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF2, self).__init__()
        self.hidden_size = hidden_size * 2 # adding the char embedding, double the hidden_size. 
        
        self.emb = layers.Embedding(model_name=model_name,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        #input_size=self.hidden_size+2 is due to we add two extra features (avg_attention) to both char embedding
        #and word embedding to boost the performance. The avg_attention is use the attention mechanism to learn 
        #a weighted average among the vectors by the model itself.
        self.enc = layers.RNNEncoder(input_size=self.hidden_size+2,
                                     hidden_size=self.hidden_size,
                                     num_layers = 1,
                                     drop_prob=drop_prob)
        
        self.highway = layers.HighwayEncoder(2, 4 * hidden_size)

        self.mod = layers.RNNEncoder(input_size=2*self.hidden_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

#         self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
       
        self.qa_outputs = nn.Linear(2*self.hidden_size , 2)

    def forward(self, char_ids, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        c_mask = (attention_mask == 1) * (token_type_ids == 1)
        q_mask = (attention_mask == 1) * (token_type_ids == 0)
        for i in range(c_mask.shape[0]):
            c_mask[i, 0] = 1
        
        c_len, q_len = attention_mask.sum(-1), q_mask.sum(-1)

        
        emb = self.emb(c=char_ids,
                       input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids
                       )
        

        enc = self.enc(emb, c_len)    # (batch_size, c_len, 2 * hidden_size)

        proj = self.highway(enc)

        mod = self.mod(enc, c_len)        # (batch_size, c_len, 2 * hidden_size)s
        
        logits = self.qa_outputs(mod)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,)
        
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs
