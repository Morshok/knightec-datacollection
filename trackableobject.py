class TrackableObject():
    def __init__(self, objectID, centroid):
            # Store objectID and initialize a list
            # of centroids using centroid
            self.objectID = objectID;
            self.centroids = [centroid];

            # Initialize a boolean to indicate whether
            # object has been counted or not
            self.hasBeenCounted = False;