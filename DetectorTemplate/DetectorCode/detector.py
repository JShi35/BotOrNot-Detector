from abc_classes import ADetector
from teams_classes import DetectionMark

class Detector(ADetector):
    def detect_bot(self, session_data):
        marked_account = []
        
        for user in session_data.users:
            confidence = 0
            bot = False

            # Feature 1: Username has 4+ digits
            username = user.get('username', '')
            digit_count = sum(c.isdigit() for c in username)
            if digit_count >= 4:
                confidence += 20

            # Feature 2: Tweet count exactly 100
            tweet_count = user.get('tweet_count', 0)
            if tweet_count == 0:
                confidence += 35

            # Feature 3: Empty description
            description = user.get('description', '')
            if not description.strip():
                confidence += 15

            # Feature 4: Empty or fictional location
            location = user.get('location', '') or '' 
            if not location.strip() or "unicorn" in location.lower():
                confidence += 15

            # Determine bot by a certain threshhold
            confidence = min(confidence, 100) 
            bot = confidence >= 50  # Threshold for marking as bot

            marked_account.append(
                DetectionMark(
                    user_id=user['id'],
                    confidence=confidence,
                    bot=bot
                )
            )
            
        return marked_account