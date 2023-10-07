## @package StableYolo
#    Copyright 2023 Hector D. Menendez
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  
#  Documentation for this module.
#
#  More details.
import random
import math
import sys
import hashlib
import numpy
import copy
import pandas as pd
from deap import creator, base, tools, algorithms
import torch
from scipy.spatial.distance import cosine
import requests                                                                      
from PIL import Image     
from statistics import mean


from ultralytics import YOLO

import logging
from bs4 import BeautifulSoup
import requests
import schedule
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionImg2ImgPipeline
import calendar
import time
import numpy as np
from io import BytesIO
import cv2
import random
import argparse


#Optimizer parameters
#numTuples = int(ConfigSectionMap("Optimizer")['numtuples'])

def int_to_binary_and_select_elements(integer, element_list):
    binary_representation = bin(integer)[2:] 
    selected_elements = []
    for i,digit in enumerate(binary_representation):
        if digit == '1':
            selected_elements.append(element_list[i])
    return selected_elements

#Parameters for the boxes
thickness=2
fontScale=0.5

model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
model.train(data='coco128.yaml', epochs=3)  # train the model
colors = np.random.randint(0, 255, size=(len(model.names), 3), dtype="uint8")

def read_box(box):
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    class_id = model.names[box.cls[0].item()]
    conf = round(box.conf[0].item(), 2)
    return [class_id, cords, conf]

def addBoxesImage(currentImage,boxesInfo):
    image = cv2.imread(currentImage)
    for box in boxesInfo:
        class_id=box[0]
        confidence=box[2]
        color = [int(c) for c in colors[list(model.names.values()).index(class_id)]]
#        color = colors[list(model.names.values()).index(class_id)]
        cv2.rectangle(image, (box[1][0], box[1][1]), (box[1][2], box[1][3]), color=color, thickness=thickness)
        text = f"{class_id}: {confidence:.2f}"
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, thickness=thickness)[0]
        text_offset_x = box[1][0]
        text_offset_y = box[1][1] - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        cv2.putText(image, text, (box[1][0], box[1][1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontScale, color=(0, 0, 0), thickness=thickness)
    cv2.imwrite(currentImage + "_yolo8.png", image)


def createNegativePrompt(selection):
    items=['illustration', 'painting', 'drawing', 'art', 'sketch','lowres', 'error', 'cropped', 'worst quality', 'low quality', 'jpeg artifacts', 'out of frame', 'watermark', 'signature']
    #integer_input =  random.randint(0,2**len(fixed_length_list)-1)
    if(selection > 2**len(items)-1):
        selection %= 2**len(items)-1
    selected_elements = int_to_binary_and_select_elements(selection, items)
    return ", ".join(selected_elements)

def createPosPrompt(prompt,selection):
    items=['photograph', 'digital','color','Ultra Real', 'film grain', 'Kodak portra 800', 'Depth of field 100mm', 'overlapping compositions', 'blended visuals', 'trending on artstation', 'award winning']
    #integer_input =  random.randint(0,2**len(fixed_length_list)-1)
    if(selection > 2**len(items)-1):
        selection %= 2**len(items)-1
    selected_elements = int_to_binary_and_select_elements(selection, items)
    return prompt + ", " + ", ".join(selected_elements)

    
def text2img(prompt,configuration={}):
    num_inference_steps = configuration['num_inference_steps']
    guidance_scale = configuration['guidance_scale']
    negative_prompt= createNegativePrompt(configuration['negative_prompt'])
    prompt= createPosPrompt(prompt,configuration['positive_prompt'])
    guidance_rescale= configuration['guidance_rescale']
    num_images_per_prompt= 4
    seed= configuration['seed']
    
    generator = torch.Generator("cuda").manual_seed(seed)
    print(prompt)
    print(negative_prompt)
    imagesAll = pipe(prompt,
                 guidance_scale=guidance_scale,
                 num_inference_steps=num_inference_steps,
                 guidance_rescale=guidance_rescale,
                 negative_prompt=negative_prompt,
                 generator=generator,num_images_per_prompt=num_images_per_prompt,
    ).images
    print(imagesAll)
    timestamp = calendar.timegm(time.gmtime())
    images=[]
    for i,image in enumerate(imagesAll):
        image.save(prompt.replace(" ", "_") + '.' +str(timestamp) +"."+str(i)+"."+ "image.png")
        images.append(prompt.replace(" ", "_") + '.' +str(timestamp)+"."+str(i)+"."+"image.png")
    return images

def img2text(image_path):
    result=model(image_path)  # predict on an image
    boxesInfo=[]
    counting={}
    for box in result[0].boxes:
        currentBox=read_box(box)
        boxesInfo.append(currentBox)
        if(currentBox[0] in counting.keys()):
            counting[currentBox[0]] += 1
        else:
            counting[currentBox[0]] = 1
    return counting,boxesInfo

class GAOptimizer:

    def __init__(self,options={},others={}):
        #GA parameters
        self.numGen = int(options['numgen'])
        self.mutProb= float(options['mut_prob'])
        self.crossProb=float(options['cross_prob'])
        self.numSel=int(options['num_sel'])
        self.muSel=int(options['mu_sel'])
        self.lambdaSel=int(options['lambda_sel'])
        self.innerMutProb=float(options['inner_mut_prob'])
        self.populationSize=int(options['population_size'])
        self.tournamentSel=int(options['tournament_sel'])
        self.weights=options["weights"]
        self.prompt=options["prompt"]

        #Individual Initialization parameters
        #self.tsize = int(options['sizetuples'])
        #self.isize = int(options['numtuples'])
        #self.types = list(options["type"+str(i)] for i in range(self.tsize))
        #self.minInt = int(options['minInt'])
        #self.maxInt = int(options['maxInt'])
        #self.minFloat= float(0)
        #self.maxFloat= float(options['noise'])

    def createElem(self):
        param_ranges_dict = {
            'num_inference_steps' : random.randint(1,100),           # Number of denoising steps
            'guidance_scale' : 20*random.uniform(0,1),                # Scale for classifier-free guidance
            'negative_prompt':random.randint(1,2**9), 
            'positive_prompt':random.randint(1,2**14),
            'guidance_rescale':random.uniform(0,1),
            'num_images_per_prompt':4,
            'seed':random.randint(1,2**9)
        }
        return param_ranges_dict

    
    ## Documentation for randomInit
    # @param icls individual composed by tuples
    # @param low vector for minimum values for each element of a tuple
    # @param top vector for maximum values of the tuples
    # @param size individual size
    # @brief this function initializes an individual
    def randomInit(self,icls):
        ind=self.createElem()
        #print(ind)
        return icls(self.createElem())

    ## Documentation for randomInit
    # @param icls individual composed by tuples
    # @param low vector for minimum values for each element of a tuple
    # @param top vector for maximum values of the tuples
    # @param size individual size
    # @brief this function initializes an individual
    def mutUniform(self,individual):
        ind2=copy.copy(individual)
        mutInd=self.createElem()
        for elem in individual.keys():
            if random.random() < self.innerMutProb:
                ind2[elem] = mutInd[elem]
        return ind2,

    def crossOverDict(self,ind1,ind2):
#        return ind1,ind2
        print("Crossing")
        cutpoint = random.randrange(1, len(ind1.keys()))
        chrom1_list = [(k, v) for k, v in ind1.items()]
        chrom2_list = [(m, n) for m, n in ind2.items()]

        offspring_1 = chrom1_list[1:cutpoint] + chrom2_list[cutpoint: len(chrom2_list)]

        offspring_2 = chrom2_list[1:cutpoint] + chrom1_list[cutpoint: len(chrom1_list)]
        print("Showing the offprint")
        print(offspring_1)
        offspring_1 = dict(offspring_1)
        offspring_2 = dict(offspring_2)
        chrom_offspring_1, chrom_offspring_2 = copy.copy(ind1), copy.copy(ind2)
        #print(chrom1.__dict__)
        #print(offspring_1.__dict__)
        chrom_offspring_1.update(offspring_1)
        chrom_offspring_2.update(offspring_2)
        print("Final offpring")
        print(chrom_offspring_1)
        print("Original")
        print(ind1)
#        chrom_offspring_1.__dict__ = offspring_1        
#        chrom_offspring_2.__dict__ = offspring_2
        return [chrom_offspring_1, chrom_offspring_2]

    
#    def fitness(self,inputs):
#        return 1
    
    def optimize(self):
#        creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
        creator.create("FitnessMax", base.Fitness, weights=self.weights)
        creator.create("Individual", dict, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("individual", self.randomInit, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evalFitness)
        toolbox.register("mate", self.crossOverDict)
        toolbox.register("mutate", self.mutUniform)
        toolbox.register("select", tools.selTournament,tournsize=self.tournamentSel)
        #The statistics for the logbook
        stats=tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg",numpy.mean,axis=0)
        stats.register("std",numpy.std,axis=0)
        stats.register("min",numpy.min,axis=0)
        stats.register("max",numpy.max,axis=0)
        #toolbox.register("elitism", tools.selBest, k=numSel)
        population = toolbox.population(n=self.populationSize)
        # The genetic algorithm, this implementation ia mu+lambda
        # it is feeded with a population of individuals, a mutation
        # and crossover probabilities and a number of generations
        offspring,logbook = algorithms.eaMuCommaLambda(population, toolbox,mu=self.muSel,lambda_=self.lambdaSel,cxpb=self.crossProb, mutpb=self.mutProb,ngen=self.numGen,stats=stats)
        # the top ten individuals are printed
        #topTen = tools.selBest(population, k=10)
        #print(topTen)
        best = tools.selBest(population, k=1)
        return best[0],offspring,logbook

    def get_caption_similarity(self, text_a, text_b):
        texts = [text_a, text_b]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # Get the embeddings
        with torch.no_grad():
            embeddings = self.modelText(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        similarity_score = 1 - cosine(embeddings[0], embeddings[1])
        return similarity_score

    def evalFitness(self,individual):
        print("Fitness")
        avgPrecision=0
        totalCount=0
        configuration={
            'num_inference_steps' : individual['num_inference_steps'],
            'guidance_scale' : individual['guidance_scale'],
            'negative_prompt':individual['negative_prompt'], 
            'positive_prompt':individual['positive_prompt'],
            'guidance_rescale':individual['guidance_rescale'],
            'seed':individual['seed']
        }

        allimages=text2img(self.prompt,configuration)
        for currentImage in allimages:
            counting,boxesInfo = img2text(currentImage)
            print(counting)
            addBoxesImage(currentImage,boxesInfo)
            for box in boxesInfo:
                totalCount+=1
                avgPrecision+=box[2]
        if(avgPrecision==0):
            return 0,
        else:
            return avgPrecision/totalCount,#individual['num_inference_steps'],
    

parser = argparse.ArgumentParser(description="Improve Image Quality")
parser.add_argument('--prompt','-p', default="Two people and a bus", help="Prompt for the input")
args = parser.parse_args()

configuration={
    'numgen': 50,
    'mut_prob': 0.2,
    'cross_prob':0.2,
    'num_sel':10,
    'mu_sel':5,
    'lambda_sel':5,
    'inner_mut_prob':0.2,
    'population_size':25,
    'tournament_sel':5,
    'weights':[1],
    'prompt':args.prompt
    }



print("Loading data")
print("GA")
gen = GAOptimizer(configuration)

sol,offspring,logbook = gen.optimize()
print("Last Generation")
print(offspring)
print("Logs")
print(logbook)
print("Best")
print(sol)
print("Done")
