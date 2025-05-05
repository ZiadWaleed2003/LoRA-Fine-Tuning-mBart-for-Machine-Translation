import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import re

"""""
    I implemented Singltone pattern here to prevent loading the model every time we create a new instance
"""
    



class Translator:
    # Class variable to track if the model has been loaded
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Translator, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.load_model()
        return cls._instance
        
    def load_model(self):
        print("Loading model... (this happens only once)")
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("ZiadWaleed/mBart-LoRA-for-MT-ar-en2")
        
        # Load PEFT config to get base model name
        config = PeftConfig.from_pretrained("ZiadWaleed/mBart-LoRA-for-MT-ar-en2")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        
        # Load LoRA adapter on top of base model
        model = PeftModel.from_pretrained(base_model, "ZiadWaleed/mBart-LoRA-for-MT-ar-en2")
        
        # Set model to evaluation mode
        model.eval()
        
        # Set up for translation
        SRC_LANG = "ar_AR"
        TGT_LANG = "en_XX"
        tokenizer.src_lang = SRC_LANG
        tokenizer.tgt_lang = TGT_LANG
        
        # Set forced BOS token to the target language
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[TGT_LANG]
        
        self.model = model.to("cpu")
        self.tokenizer = tokenizer
    
    def clean_arabic(self, text):
        # Remove newlines
        text = re.sub(r'\\n|\n', ' ', text)
        # Normalize 'أ', 'إ', 'آ' -> 'ا'
        text = re.sub(r'[أإآ]', 'ا', text)
        # Normalize 'ة' -> 'ه'
        text = re.sub(r'[ة]', 'ه', text)
        # Normalize Eastern Arabic numerals to Western Arabic numerals
        arabic_numerals = '٠١٢٣٤٥٦٧٨٩'
        western_numerals = '0123456789'
        trans = str.maketrans(arabic_numerals, western_numerals)
        text = text.translate(trans)
        # Remove diacritics (tashkeel)
        text = re.sub(r'[\u064B-\u065F]', '', text)
        # Remove tatweel
        text = re.sub(r'\u0640', '', text)
        # Remove Arabic and Western punctuation
        text = re.sub(r'[،؛«»,!?()\[\]{}"\'\\]', '', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def normalize_contractions(self, text):
        # Stage 1: Fix compressed forms (im → I'm → I am)
        text = re.sub(r"\bim\b", "I'm", text)  # Catch "im" first
        text = re.sub(r"\bive\b", "I've", text)  # Example: "ive" → "I've"
        
        # Stage 2: Standard contraction expansion
        contractions = {
            "don't": "do not", "isn't": "is not", 
            "I'm": "I am", "you're": "you are",  # Now handles "I'm" → "I am"
        }

        for cont, expanded in contractions.items():
            text = text.replace(cont, expanded)
        return text
    
    def translate(self, arabic_text):
        arabic_text = self.clean_arabic(arabic_text)
        inputs = self.tokenizer(arabic_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}  # Ensure on CPU
        
        with torch.no_grad():
            translated_tokens = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length = 256,
                num_beams=5,
                early_stopping=True,
            )
            
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        translated_text = self.normalize_contractions(translated_text)
        
        return translated_text