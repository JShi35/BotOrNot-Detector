from abc_classes import ADetector
from teams_classes import DetectionMark

class Detector(ADetector):
    def detect_bot(self, session_data):
        marked_account = []
        for user in session_data.users:
            z_score = user['z_score']
            # Simple threshold-based logic
            if z_score > 2.0:
                confidence = 80  # High confidence for bots
                bot = True
            elif z_score > 1.0:
                confidence = 60  # Moderate confidence
                bot = True
            else:
                confidence = 30  # Low confidence (likely human)
                bot = False
            marked_account.append(
                DetectionMark(
                    user_id=user['id'],
                    confidence=confidence,
                    bot=bot
                )
            )
        return marked_account