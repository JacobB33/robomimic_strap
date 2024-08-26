import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, pipeline, AutoTokenizer, CLIPTextModelWithProjection

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



class MiniLMLangEncoder:
    def __init__(self, device):
        # TODO: not supported yet bc transformer architecture breaks with smaller embedding size :/
        os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
        model_variant = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = device
        self.lang_model = AutoModel.from_pretrained(model_variant).to(device).eval()

        self.tz = AutoTokenizer.from_pretrained(model_variant, TOKENIZERS_PARALLELISM=True)

    def get_lang_emb(self, lang):
        if lang is None:
            return None
        
        # Tokenize sentences
        encoded_input = self.tz(lang, padding=True, truncation=True, return_tensors='pt').to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.lang_model(**encoded_input)

        # Perform pooling
        lang_embed = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        lang_embed = F.normalize(lang_embed, p=2, dim=1)
        
        if isinstance(lang, str):
            lang_emb = lang_emb[0]

        return lang_embed

class CLIPLangEncoder:
    def __init__(self, device):
        os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
        model_variant = "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32"
        self.device = device
        try:
            self.lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
                model_variant,
                cache_dir=os.path.expanduser("~/tmp/clip")
            ).to(device).eval()
        except EnvironmentError:
            # use proxy on strip district cluster
            self.lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
                model_variant,
                cache_dir=os.path.expanduser("~/tmp/clip"),
                proxies={'http': 'http://rb-proxy-sl.bosch.com:8080'}
            ).to(device).eval()

        try:
            self.tz = AutoTokenizer.from_pretrained(model_variant, TOKENIZERS_PARALLELISM=True)
        except EnvironmentError:
            # use proxy on strip district cluster
            self.tz = AutoTokenizer.from_pretrained(model_variant, TOKENIZERS_PARALLELISM=True, proxies={'http': 'rb-proxy-sl.bosch.com:8080'})

    def get_lang_emb(self, lang):
        if lang is None:
            return None
            # Convert single string to list for consistent processing
        if isinstance(lang, str):
            lang = [lang]
        batch_size = 100
        all_lang_embs = []
        # self.print_cuda_memory_gb()
        # print(f"the length of the lang to embed is {len(lang)}") 
        for i in range(0, len(lang), batch_size):
            batch = lang[i:i+batch_size]
        
            with torch.no_grad():
                tokens = self.tz(
                    text=batch,                   # batch of sentences to be encoded
                    add_special_tokens=True,      # Add [CLS] and [SEP]
                    padding="max_length",
                    return_attention_mask=True,   # Generate the attention mask
                    return_tensors="pt",          # ask the function to return PyTorch tensors
                    ).to(self.device)

                lang_embs = self.lang_emb_model(**tokens)['text_embeds'].detach()
                all_lang_embs.append(lang_embs)
    
        # Concatenate all batches
        all_lang_embs = torch.cat(all_lang_embs, dim=0)

        # If input was a single string, return a single embedding
        if len(lang) == 1:
            return all_lang_embs[0]
    
        return all_lang_embs

