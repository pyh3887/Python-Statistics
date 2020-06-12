from django.contrib import admin
from myapp.models import SurveyData

# Register your models here.

class SurveyDataAdmin(admin.ModelAdmin):
    list_display = ('id','job','gender','game_time')

admin.site.register(SurveyData,SurveyDataAdmin)