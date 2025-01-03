import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, DeiTModel
from config import Config
from utils.arg_parser import get_args
from utils.data_processing import preprocess_data

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

### Image features extraction model
class ImageEmbedding(nn.Module):
    def __init__(self):
        super(ImageEmbedding, self).__init__()
        self.process = AutoImageProcessor.from_pretrained(Config.image_model)
        self.model = DeiTModel.from_pretrained(Config.image_model)
        #self.model = nn.Sequential(*list(self.model.children())[:3])
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, image_ids):
        inputs = self.process(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs.to(device))
            
        image_embedding = outputs.last_hidden_state
        return image_embedding, image_ids
    
### Quesion embedding model
class QuesEmbedding(nn.Module):
    def __init__(self, input_size=768, output_size=768):
        super(QuesEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.textmodel_dir)
        self.phobert = AutoModel.from_pretrained(Config.textmodel_dir)
        self.lstm = nn.LSTM(input_size, output_size, batch_first=True)

    def forward(self, ques):
        tokenized_input = self.tokenizer(ques, return_tensors='pt', padding='max_length', max_length=Config.MAX_LEN_QUES, truncation=True)
        ques = self.phobert(**tokenized_input.to(device)).last_hidden_state
        _, (h, _) = self.lstm(ques)
        return h.squeeze(0)
    
### Answer embedding model
class AnsEmbedding(nn.Module):
    def __init__(self, input_size=768):
        super(AnsEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.textmodel_dir)
        self.phobert_embed = AutoModel.from_pretrained(Config.textmodel_dir).embeddings.to(device)

    def forward(self, ans):   
        tokenized_input = self.tokenizer(ans, return_tensors='pt', padding='max_length', max_length=Config.MAX_LEN_ANS, truncation=True, return_attention_mask=False)
        ans = self.phobert_embed(**tokenized_input.to(device))
        return tokenized_input['input_ids'], ans