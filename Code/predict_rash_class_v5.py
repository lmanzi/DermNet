#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# PRELIMINARIES
  
# Import packages
from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np
import pandas as pd

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

# Load model
model = load_model('/Users/jamesmanzi/Desktop/Personal/WIS Work/Lincoln/Sophomore/Project/VGG19test.h5')


# Set prediction vector index to label lookup
index_class_dict = {0: 'Acne',
 1: 'Eczema',
 2: 'Scabies',
 3: 'Ringworm'}



# Set target image size
image_size = 224


# Import ground truth data for images
demo_lists_df_curated = pd.read_csv('/Users/jamesmanzi/Desktop/Personal/WIS Work/Lincoln/Sophomore/Project/demo_lists_df_curated.csv')
ground_truth_df = demo_lists_df_curated[['dir_seq', 'ground_truth']]
ground_truth_df.columns = ['image_id', 'ground_truth']
ground_truth_df['image_id'] = ground_truth_df['image_id'].astype(str)

# Set first and last image numbers
min_image_num = 1
max_image_num = len(ground_truth_df['image_id'])

# Convert to valid input list
valid_image_nums = list(range(min_image_num, max_image_num+1))
valid_image_nums = [str(item) for item in valid_image_nums]
valid_image_nums.append('quit')


# Define function to convert prediction probability to confidence label
def get_confidence(prob):
    
    if prob < 0.5:
        confidence = 'Low'
        
    elif prob >= 0.7:
        confidence = 'High'
        
    else:
        confidence = 'Medium'

    return confidence


# Define function to get user selection, score and report results
def demo_function(response):
    
     # Import image and reshape to match input rquirements of model
      
      # Convert to image path
    img_path = '/Users/jamesmanzi/Desktop/Demo_Images_2/' + str(response) + '.jpg'
     
      # Load image
    img = load_img(img_path)
    
      # Re-size image
    img = img.resize((image_size,image_size))
      
      # Convert to numpy array
    img = np.array(img)
    
      # Divide all values by constant
    img = img / 255.0
    
      # Reshape
    img = img.reshape(1,image_size,image_size,3)
    
    
    # Apply neural network model
    model_pred = model.predict(img)
    
    
    # Convert model output to label and probability
    
      # Get index of maximum prediction
    pred_class_index = np.argmax(model_pred)
    
      # Use dictionary to lookup name of this class
    pred_class = index_class_dict[pred_class_index]
    
      # Extract maximum prediction probability
    pred_prob = np.amax(model_pred)
    
      # Convert prob to confidence
    confidence =  get_confidence(pred_prob)  
    
    
    # Get ground truth
    ground_truth = list(ground_truth_df[ground_truth_df['image_id'] == response]['ground_truth'])[0]
    
     
    
    # Creat labels for image
 
    line_1 = 'Model Prediction: ' + pred_class
    line_2 = 'Confidence: ' + confidence
    line_3 = 'Actual Condition: ' + ground_truth
    
    
    image_text = line_1 + '\n' + line_2 + '\n' + '\n' + line_3


    # Print labeled image
    
      # Load image for image processing program
    img_proc = Image.open(img_path)
    
      # Create image object that can include text
    draw = ImageDraw.Draw(img_proc)
    
      # Set the font
    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 40)
    
      # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((30, 30), image_text,  font = fnt, fill = (0,0,255))
    
      # Display image
    img_proc.show()







    








