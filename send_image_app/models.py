from django.db import models
from datetime import date
from django.utils.timezone import now

# class Image(models.Model):
#   id = models.AutoField(primary_key=True)
#   result = models.IntegerField(blank=True, null=True)
#   proba = models.FloatField(default=0.0)
#   registered_date = models.DateField(default=now)
  
#   def __str__(self):
#     if self.proba == 0.0:
#       return '%s' % (self.registered_date.strftime('%Y-%m-%d'))
#     else:
#       return '%s, %d' % (self.registered_date.strftime('%Y-%m-%d'), self.result)
    
class ModelFile(models.Model):
  id = models.AutoField(primary_key=True)
  result = models.IntegerField(blank=True, null=True)
  proba = models.FloatField(default=0.0)
  registered_date = models.DateField(default=now)
  image = models.ImageField(upload_to='documents/')
  
  def __str__(self):
    if self.proba == 0.0:
      return '%s' % (self.registered_date.strftime('%Y-%m-%d'))
    else:
      return '%s, %d' % (self.registered_date.strftime('%Y-%m-%d'), self.result)