from django.db import models


class Post(models.Model):
    name = models.CharField(max_length=200)
    image = models.ImageField(upload_to="images/", null=True)
    created_on = models.DateTimeField(auto_now_add=True, null=True)

    def __str__(self):
        return self.name
