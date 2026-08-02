from django.db import models
from django.contrib.auth.models import User

class PostureSession(models.Model):
    # If you add login later, we can link it to a user. For now, it can be blank.
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    total_duration = models.IntegerField(default=0)  # in seconds
    slouch_duration = models.IntegerField(default=0) # in seconds
    posture_score = models.IntegerField(default=100) # percentage 0-100

    def __str__(self):
        return f"Session on {self.timestamp.strftime('%Y-%m-%d %H:%M')} - Score: {self.posture_score}%"