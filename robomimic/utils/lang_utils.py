import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, pipeline, AutoTokenizer, CLIPTextModelWithProjection

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class DistilBERTLangEncoder:
    def __init__(self, device):
        os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
        # https://arxiv.org/abs/1910.01108 -> inspired by Yell At Your Robot
        model_variant = "distilbert-base-uncased"
        self.device = device
        self.lang_model = AutoModel.from_pretrained(model_variant).to(device).eval()

        self.tz = AutoTokenizer.from_pretrained(model_variant, TOKENIZERS_PARALLELISM=True)

    def get_lang_emb(self, lang):
        print(lang)
        if lang is None:
            return None
            
        was_str = False
        if isinstance(lang, str):
            lang = [lang]
            was_str = True
        
        # Tokenize sentences
        inputs = self.tz(lang, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # Compute token embeddings
        with torch.no_grad():
            lang_embed = self.lang_model(**inputs).last_hidden_state.mean(dim=1)
        
        # If input was a single string, return a single embedding
        if len(lang) == 1 and was_str:
            return lang_embed[0]

        return lang_embed
        
class MiniLMLangEncoder:
    def __init__(self, device):
        # TODO: not supported yet bc transformer architecture breaks with smaller embedding size :/
        os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
        # https://arxiv.org/abs/2002.10957 -> inspired by BAKU
        model_variant = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = device
        self.lang_model = AutoModel.from_pretrained(model_variant).to(device).eval()

        self.tz = AutoTokenizer.from_pretrained(model_variant, TOKENIZERS_PARALLELISM=True)

    def get_lang_emb(self, lang):
        # BAKU usage https://github.com/siddhanthaldar/BAKU/blob/80bc41b43300f547bcbb3b7dd91e276b1dfa83b0/baku/data_generation/generate_libero.py#L24
        # Huggingface refence of SentenceTransformer equivalence https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        if lang is None:
            return None
            
        was_str = False
        if isinstance(lang, str):
            lang = [lang]
            was_str = True
        
        # Tokenize sentences
        encoded_input = self.tz(lang, padding=True, truncation=True, return_tensors='pt').to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.lang_model(**encoded_input)

        # Perform pooling
        lang_embed = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        lang_embed = F.normalize(lang_embed, p=2, dim=1)
        
        # If input was a single string, return a single embedding
        if len(lang) == 1 and was_str:
            return lang_embed[0]

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
        print(lang)
        if lang is None:
            return None
            
        was_str = False
        if isinstance(lang, str):
            lang = [lang]
            was_str = True
        
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
        if len(lang) == 1 and was_str:
            return all_lang_embs[0]
    
        return all_lang_embs

