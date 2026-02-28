
import torch
import torch.nn as nn
import torch.nn.functional as F
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
                
class TextFeature(nn.Module):
    def __init__(self, model = 'conch_ViT-B-16'):
        super(TextFeature, self).__init__()

        model_cfg = model
        checkpoint_path = './model/pytorch_model_CONCH.bin'
        self.model, self.preprocess = create_model_from_pretrained(model_cfg, checkpoint_path)
        self.tokenizer = get_tokenizer() # load tokenizer
        self.max_length = 77*100-60
        self.model.cuda()

    def token(self, x):
        text_tokens = tokenize(texts=x, tokenizer=self.tokenizer) # tokenize the text
        return text_tokens

    def forward(self, x):
         with torch.no_grad():
            batch_size, seq_length = x.shape
            if seq_length > self.max_length:

                feature_list = []
                for i in range(batch_size):
                    sample_tokens = x[i:i+1, :] 
                    
                    if sample_tokens.size(1) > self.max_length:
                        chunk_features = []
                        for j in range(0, sample_tokens.size(1), self.max_length):
                            chunk = sample_tokens[:, j:j+self.max_length]
                            if chunk.size(1) < self.max_length:
                                padding_size = self.max_length - chunk.size(1)
                                chunk = torch.cat([chunk, torch.zeros(chunk.size(0), padding_size, dtype=chunk.dtype, device=chunk.device)], dim=1)
                            
                            chunk_feature = self.model.encode_text(chunk)
                            chunk_features.append(chunk_feature)
                        features = torch.cat(chunk_features, dim=1)
                    feature_list.append(features)  
                text_features = torch.cat(feature_list, dim=0)
            else:
                text_features = self.model.encode_text(x)

            return text_features
                
if __name__ == '__main__':
    TextFeature()

