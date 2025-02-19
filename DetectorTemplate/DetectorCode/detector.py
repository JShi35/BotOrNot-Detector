import re
from abc_classes import ADetector
from teams_classes import DetectionMark

class Detector(ADetector):
    def detect_bot(self, session_data):
        
        """
        Modified:
          - Removes the '0 tweets => +35' feature
          - Uses Jaccard similarity on post text
        """

        # 1) Build a map of user_id -> list of post texts (if session_data has posts)
        user_posts_map = {}
        if hasattr(session_data, 'posts'):
            for post in session_data.posts:
                author_id = post.get('author_id')
                text = post.get('text', '')
                if author_id not in user_posts_map:
                    user_posts_map[author_id] = []
                user_posts_map[author_id].append(text)

        # 2) function for Jaccard similarity
        def jaccard_similarity(str1, str2):
            """
            Compute Jaccard similarity between two strings:
            intersection(words) / union(words)
            """
            set1 = set(str1.lower().split())
            set2 = set(str2.lower().split())
            if not set1 or not set2:
                return 0.0
            intersection_size = len(set1.intersection(set2))
            union_size = len(set1.union(set2))
            if union_size == 0:
                return 0.0
            return intersection_size / union_size

        marked_account = []

        for user in session_data.users:
            confidence = 0
            bot = False

            # ====================
            # ACCOUNT-LEVEL CHECKS
            # ====================
            username = user.get('username', '')
            description = user.get('description', '')
            location = user.get('location', '') or ''
            tweet_count = user.get('tweet_count', 0)

            # Feature 1: Username has 4+ digits
            if re.search(r'\d{4,}', username):
                confidence += 20

            # Feature 2: Very low tweet count (<5)
            if tweet_count < 5:
                confidence += 10

            # Feature 3: Empty description
            if not description.strip():
                confidence += 15

            # Feature 4: Empty or suspicious location
            suspicious_locations = ["unicorn", "nowhere", "unknown", "mars", "universe", "hell"]
            loc_lower = location.lower()
            if not location.strip() or any(loc in loc_lower for loc in suspicious_locations):
                confidence += 15

            # ====================
            # POST-LEVEL CHECKS
            # ====================
            user_id = user.get('id')
            user_posts = user_posts_map.get(user_id, [])
            total_posts = len(user_posts)

            if total_posts > 0:
                total_words = 0
                total_hashtags = 0
                total_mentions = 0
                unique_texts = set()

                for text in user_posts:
                    words = text.split()
                    total_words += len(words)
                    total_hashtags += text.count("#")
                    total_mentions += text.count("@")
                    unique_texts.add(text)

                # Hashtag ratio
                if total_words > 0:
                    hashtag_ratio = total_hashtags / total_words
                    if hashtag_ratio > 0.3:
                        confidence += 20

                # Excessive mentions
                if total_mentions > 10:
                    confidence += 10  # moderate weight

                # Repetitive short phrases (duplicate posts)
                if total_posts > 5:
                    unique_ratio = len(unique_texts) / total_posts
                    if unique_ratio < 0.5:  # more than half are duplicates
                        confidence += 20

                # check for "spam phrases"
                spam_phrases = [
                    "check this out!",
                    "follow me!",
                    "click the link!",
                    "you won't believe this!",
                    "win a free"
                ]
                for text in user_posts:
                    lower_text = text.lower()
                    if any(phrase in lower_text for phrase in spam_phrases):
                        confidence += 20
                        break      # only once per user

                # Jaccard Similarity Check:
                # If any pair of posts is highly similar (>0.7), add confidence once.
                similar_posts_found = False
                if total_posts > 1:
                    
                    for i in range(total_posts):
                        for j in range(i+1, total_posts):
                            sim = jaccard_similarity(user_posts[i], user_posts[j])
                            if sim > 0.7:
                                confidence += 20
                                similar_posts_found = True
                                break
                        if similar_posts_found:
                            break

            confidence = min(confidence, 100)

            bot_threshold = 50
            bot = (confidence >= bot_threshold)

            marked_account.append(
                DetectionMark(
                    user_id=user['id'],
                    confidence=confidence,
                    bot=bot
                )
            )
            
        return marked_account
