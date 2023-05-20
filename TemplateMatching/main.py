# library imports 
import os

# modules
from ImageProcessor import ImageProcessor
from utils import Match, CONFIG, suffix, printDivider

def workflowOne(images):
    # remember to change FILTER to Filter.MANUAL in CONFIG
    for image in images:
        processImg = ImageProcessor(image)
        processImg.match(True, Match.TWO_D)
    for image in images:
        processImg = ImageProcessor(image)
        processImg.manual(True, Match.TWO_D)
    for image in images:
        processImg = ImageProcessor(image)
        processImg.filter(True)
        processImg.fitProject(True)
    ImageProcessor.plotResult(True)

if __name__ == "__main__":
    images = [file for file in os.listdir(os.path.join(os.getcwd(),"img")) if suffix(file) in CONFIG.FILE_TYPES]
    workflowOne(images[0:2])
    