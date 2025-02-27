import re
import os
import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from transformers import BertTokenizer, BertModel

from abc_classes import ADetector
from teams_classes import DetectionMark

class Detector(ADetector):
    def __init__(self):
        super().__init__()
        
        # 1) Load your XGBoost model (trained offline)
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "model.xgb")

        self.clf = xgb.XGBClassifier()
        self.clf.load_model(model_path) 
        
        # 2) Load BERT model & tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        
        # 3) adjust the threshold for classification
        self.threshold = 0.5
    
    def detect_bot(self, session_data):
        """
          - Extract structured features for each user.
          - Embedding user description and posts with bert.
          - Combine features into one vector.
          - Use XGBoost to get prob(bot).
          - Build a DetectionMark with confidence=int(prob*100).
        """
        
        # Build a map: user_id -> list of post texts
        user_posts_map = {}
        if hasattr(session_data, 'posts'):
            for post in session_data.posts:
                author_id = post.get('author_id')
                text = post.get('text', '')
                if author_id not in user_posts_map:
                    user_posts_map[author_id] = []
                user_posts_map[author_id].append(text)

        marked_account = []
        
        for user in session_data.users:
            user_id = user.get('user_id') or user.get('id')
            username = user.get('username', '')
            description = user.get('description', '')
            location = user.get('location', '') or ''
            tweet_count = int(user.get('tweet_count', 0))
            
            # (1) STRUCTURED FEATURES
            username_digits = sum(c.isdigit() for c in username)
            username_length = len(username)
            description_length = len(description)
            has_empty_description = 1 if description_length == 0 else 0
            loc_lower = location.strip().lower()
            suspicious_locs = {"nowhere","unknown","unicorn","mars","universe","hell"}
            has_suspicious_location = 1 if (not loc_lower or loc_lower in suspicious_locs) else 0
            
            # simple post-level aggregates
            user_posts = user_posts_map.get(user_id, [])
            if len(user_posts) > 0:
                avg_post_length = np.mean([len(p) for p in user_posts])
                # Duplicate ratio
                dup_count = sum(pd.Series(user_posts).duplicated()) if len(user_posts) > 1 else 0
                duplicate_post_ratio = dup_count / len(user_posts) if len(user_posts) > 1 else 0
                total_words = sum(len(p.split()) for p in user_posts)
                total_hashtags = sum(p.count("#") for p in user_posts)
                total_mentions = sum(p.count("@") for p in user_posts)
                hashtag_ratio = (total_hashtags / total_words) if total_words>0 else 0
                mention_ratio = (total_mentions / total_words) if total_words>0 else 0
                url_ratio = np.mean([1 if re.search(r"https?://", p.lower()) else 0 for p in user_posts])
            else:
                avg_post_length = 0.0
                duplicate_post_ratio = 0.0
                hashtag_ratio = 0.0
                mention_ratio = 0.0
                url_ratio = 0.0
            
            # (2) BERT EMBEDDINGS (description & posts)
            desc_emb = self.get_bert_embedding(description)
            post_emb = self.embed_user_posts(user_posts)
            
            # (3) Combine all features            
            structured_array = np.array([
                username_digits, username_length, description_length,
                has_empty_description, has_suspicious_location, tweet_count,
                avg_post_length, duplicate_post_ratio, hashtag_ratio,
                mention_ratio, url_ratio
            ], dtype=np.float32)
            
            # final feature vector
            features = np.hstack([structured_array, desc_emb, post_emb]).reshape(1, -1)
            
            # (4) Predict probability of bot
            probs = self.clf.predict_proba(features)[0]  # [prob_class0, prob_class1]
            prob_bot = probs[1]
            is_bot = (prob_bot >= self.threshold)
            
            # (5) Build the detectionmark
            detection_mark = DetectionMark(
                user_id=user_id,
                confidence=int(prob_bot*100),
                bot=is_bot
            )
            
            marked_account.append(detection_mark)
        
        return marked_account
    
    # Helpers
    def get_bert_embedding(self, text, max_tokens=256):
        """Embed text with BERT (mean pooling). Returns a 768-d vector."""
        text = text.strip()
        if not text:
            return np.zeros(768, dtype=np.float32)
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return emb.astype(np.float32)
    
    def embed_user_posts(self, user_posts, max_tokens=256):
        """Average BERT embeddings across all user posts."""
        if not user_posts:
            return np.zeros(768, dtype=np.float32)
        embeddings = []
        for txt in user_posts:
            emb = self.get_bert_embedding(txt, max_tokens)
            embeddings.append(emb)
        return np.mean(embeddings, axis=0).astype(np.float32)
