import enum

class Direction(enum.Enum):
    Down = -1,
    Still = 0,
    Up = 1

class TrackableObject():
    def __init__(self, objectID, centroid):
            # Store objectID and initialize a list
            # of centroids using centroid
            self.objectID = objectID;
            self.centroids = [centroid];

            # Initialize a boolean to indicate whether
            # object has been counted or not
            self.trackingDirection = Direction.Still;
            self.hasBeenCounted = False;