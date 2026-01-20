from PIL import Image
import cv2
import os
from datetime import datetime
import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
# from langchain.pydantic_v1 import BaseModel, Field
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from typing import Optional, Type


cap  = cv2.VideoCapture(0)
# Load BLIP model and processor
processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

class SearchInput(BaseModel):
    query: str = Field(description="The search query")

class SearchTool(BaseTool):
    name:str = "search"
    description:str  = "Search for information on the web"
    args_schema:Type[BaseModel] = SearchInput
    
    def _run(self, query):
        # You would typically use a search API like Serper or SerpAPI
        # This is a simplified example
        search_api_key = os.getenv("SEARCH_API_KEY")
        
        try:
            # Placeholder for actual search API call
            # In a real implementation, you would use:
            # - SerpAPI
            # - Google Custom Search API
            # - Bing Search API
            # - etc.
            
            # Example with a fake response:
            return f"Here are search results for '{query}': [Simplified search results would appear here]"
        except Exception as e:
            return f"Error performing search: {str(e)}"
        
class TimeZoneInput(BaseModel):
    timezone: str = Field(description="The timezone to get current time for, defaults to local")

class TimeTool(BaseTool):
    name:str  = "get_time"
    description:str  = "Get the current time, optionally for a specific timezone"
    args_schema:Type[BaseModel]  = TimeZoneInput
    
    def _run(self, timezone="local") -> str:
        try:
            if timezone.lower() == "local":
                current_time = datetime.now().strftime("%H:%M:%S")
                return f"The current local time is {current_time}"
            else:
                # For simplicity, just return local time
                # In a real app, you'd use pytz or similar to handle timezones
                current_time = datetime.now().strftime("%H:%M:%S")
                return f"The current time is {current_time} (Note: timezone functionality is simplified)"
        except Exception as e:
            return f"Error getting time: {str(e)}"

class WeatherInput(BaseModel):
    location: str = Field(description="The city and state/country to get weather for")

class WeatherTool(BaseTool):
    name:str  = "get_weather"
    description:str  = "Get the current weather for a location"
    args_schema:Type[BaseModel] = WeatherInput
    
    def _run(self, location:str) -> str:
        # You would typically use a weather API like OpenWeatherMap
        # This is a simplified example
        api_key = os.getenv("WEATHER_API_KEY")
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        
        params = {
            "q": location,
            "appid": api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                temp = data["main"]["temp"]
                description = data["weather"][0]["description"]
                return f"The weather in {location} is {description} with a temperature of {temp}°C"
            else:
                return f"Error getting weather: {data.get('message', 'Unknown error')}"
        except Exception as e:
            return f"Failed to get weather information: {str(e)}"

class CapDescTool(BaseTool):     
    name:str ="describe_view_from_camera",
    description:str ="Capture an image from the camera and describe what is seen."
    
    def _run(self) -> str:
        """Capture an image from camera and describe what is seen."""
        if cap.isOpened():
            flag, img = cap.read()
            if flag:
                cv_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image
                pil_image = Image.fromarray(cv_image)

                return self.predict(pil_image)
        return "Image can not be captured!"

    def predict(self, pil_image):
        # Preprocess and generate caption
        inputs = processor(
            images=pil_image, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output[0], skip_special_tokens=True)
        print(caption, type(caption))
        return caption
