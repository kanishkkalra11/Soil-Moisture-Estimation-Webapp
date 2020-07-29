from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from count_moisture_content import EqualMeanGreyValueCounter
from img_tools import show_image, show_images
import time, os, cv2, sys
from django.contrib.auth.decorators import login_required

# Create your views here.
@login_required
def home(request):
	return render(request, 'app/home.html')

@csrf_exempt
def process(request):

	try:
		if request.method == 'POST' and request.FILES['soil_img']:
			soil_type = request.POST.get("soil_type")
			print(soil_type)
			if (soil_type==None):
				return HttpResponse("Soil Type not selected! Go back and try again.")

			myfile = request.FILES['soil_img']
			fs = FileSystemStorage()
			filename = fs.save("soil_img.jpg", myfile)
			img_url = fs.url(filename)
			time.sleep(1)
			start_time = time.time()
			image = cv2.imread('media/{}'.format(filename), cv2.IMREAD_COLOR)
			res = EqualMeanGreyValueCounter(image)

			grey = res.mean_of_eq_mean_grey_values
			if(soil_type=="Clayey"):
				moisture = int(round(0.0424*((grey)**2) - 14.48*grey + 1244.8))
			elif(soil_type=="Ash"):
				moisture = int(round(0.1333*((grey)**2) - 39.462*grey + 2927.1))
			elif(soil_type=="Clayey_Sand"):
				moisture = int(round(-0.0008*((grey)**2) - 0.458*grey + 82.684))
			elif(soil_type=="Sandy"):
				moisture = int(round(-0.0008*((grey)**2) - 0.1944*grey + 46.127))
			return render(request, 'app/display.html', {'Moisture':moisture, 'Image':img_url})
	except Exception as e:
		print(e)
		return HttpResponse("No image uploaded! Go back and try again.")





