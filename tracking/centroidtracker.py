# Import necessary packages
from ast import Or
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

# Define CentroidTracker class
class CentroidTracker():

    # Initialize the CentroidTracker class
    def __init__(self, maxDisappeared=50):
        # Initialize next unique object ID
        # along with two OrderDict() to keep
        # track of objects in and out of frame
        self.nextObjectID = 0;
        self.objects = OrderedDict();
        self.disappeared = OrderedDict();

        # Maximum number of consecutive frames
        # a given object is allowed to be out 
        # of frame before deregistering it
        # from object tracking
        self.maxDisappeared = maxDisappeared;

    def register(self, centroid):
        # Store an object centroid with
        # the next available object ID
        self.objects[self.nextObjectID] = centroid;
        self.disappeared[self.nextObjectID] = 0;
        self.nextObjectID += 1;
    
    def deregister(self, objectID):
        # Delete an object from our
        # dictionaries given an object ID
        del self.objects[objectID];
        del self.disappeared[objectID];

    def update(self, rects):
        # Check if list of input bounding
        # boxes is empty
        if len(rects) == 0:
            # Loop through all existing tracked 
            # objects and mark them as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1;
                
                # If object has been missing for more than 
                # maxDisappear number of frames, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID);
            
            # return early, no centroids or tracking
            # information to update
            return self.objects;
        
        # Initialize an array of input centroids
        # for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int");

        # Loop through the input 
        # bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # Use the bounding box coordinates to 
            # derive the centroid
            centerX = int((startX + endX) / 2.0);
            centerY = int((startY + endY) / 2.0);
            inputCentroids[i] = (centerX, centerY);
        
        # If no objects are currently being tracked,
        # register each of the input centroids
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i]);
        # Else currently tracked objects have to be 
        # matched with the input centroids
        else:
            # Fetch the set of object IDs and 
            # corresponding centroids
            objectIDs = list(self.objects.keys());
            objectCentroids = list(self.objects.values())

            # Compute distance between each pair of 
            # object centroid and input centroid in
            # an attempt to match them
            distances = dist.cdist(np.array(objectCentroids), inputCentroids);

            # Find smallest value in each row and 
            # sort row indexes based on their minimum
            # values to move row with smallest value to
            # the *front* of the index list
            rows = distances.min(axis=1).argsort();

            # Perform similar process on the columns 
            # by finding smallest value in each column
            # and sorting using previously computed row
            # index list
            columns = distances.argmin(axis=1)[rows];

            # Keep track of rows and column indices already
            # examined in order to determine whether or not 
            # to update, register or deregister an object
            usedRows = set();
            usedColumns = set();

            # Loop through the combination of
            # (row, column) index tuples
            for (row, column) in zip(rows, columns):
                # If row or column has already been
                # examined, ignore index tuple
                if row in usedRows or column in usedColumns:
                    continue;

                # Otherwise, get objectID for current row,
                # set its ned centroid, and reset disappeared counter
                objectID = objectIDs[row];
                self.objects[objectID] = inputCentroids[column];
                self.disappeared[objectID] = 0;

                # Indicate that each of the row and column 
                # indices have been examined respectively
                usedRows.add(row);
                usedColumns.add(column);

            # Compute row and column indices that have not
            # yet been examined
            unusedRows = set(range(0, distances.shape[0])).difference(usedRows);
            unusedColumns = set(range(0, distances.shape[0])).difference(usedColumns);

            # In the event that number of object centroids is
            # equal to or greater than the number of input 
            # centroids, we need to check if some of these objects
            # have disappeared
            if distances.shape[0] >= distances.shape[1]:
                # Loop through unused row indices
                for row in unusedRows:
                    # Fetch objectID for corresponding
                    # row index and increment disappeared counter
                    objectID = objectIDs[row];
                    self.disappeared[objectID] += 1;

                    # Check if number of consecutive frames object
                    # has been missing warrants a deregister
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID);
            # Otherwise we need to register each new input 
            # centroid as a trackable object
            else:
                for column in unusedColumns:
                    self.register(inputCentroids[column]);

        # Return the set of trackable objects
        return self.objects;