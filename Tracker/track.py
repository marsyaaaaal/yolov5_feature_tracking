
import math
import numpy

class TrackState:

    Tentative = 1  # tracks registration stage stage, initialized but not confirmed
    Confirmed = 2  # Confirmed tracks, tracks that is done in n_init and registraion
    Deleted = 3  # removed tracks


class Tracks:
    def __init__(self, hash, xyxy, id, class_id, n_init, wh, max_age, xy, conf):
        """Tracks for each vehicle"""
        self.track_id = id
        self.xyxy = xyxy
        self.class_id = class_id
        self.n_init = n_init
        self.track_state = TrackState.Tentative
        self.calls = 0
        self.wh = wh
        self.hash = hash
        self.last_seen = 0
        self.max_age = max_age
        self.missed = False
        self.thresh = self.computeEucDist(xyxy, self.wh, False)
        self.base_thresh = self.computeEucDist(xyxy, self.wh, True)
        self.base_xy = xy
        self.is_base_changed = False
        self.conf = conf

    def computeEucDist(self, xyxy, wh, for_base):
        """compute for treshold using euclidean distance"""
        x1, y1, x2, y2 = xyxy

        x = (x1 + x2)/2
        y = (y1 + y2)/2

        if for_base:
            a = wh[0]/6 if wh[0]<wh[1] else wh[1]/6
        else:
            a = wh[0]/2 if wh[0]<wh[1] else wh[1]/2
        xy = numpy.array((x, y))
        u = numpy.array((x+a, y+a))

        dist = numpy.sqrt(((u[0]-xy[0])**2)+((u[1]-xy[1])**2))

        return dist
    
    def distance(self, p1, p2):
        """Compute for euclidean distance"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def update(self, xyxy, hash, wh, class_id, xy, match, conf):
        """Keeps each track updated"""
        if match > 0.95:
            self.hash = hash
            
        self.thresh = self.computeEucDist(xyxy, wh, False)
        if self.calls == self.n_init:
            self.track_state = TrackState.Confirmed
            # print(f'vehicle id = {self.track_id} has been registered')
            self.hash = hash
            self.base_xy = xy
            self.base_thresh = self.computeEucDist(xyxy, self.wh, True)
            
            
        elif self.calls < self.n_init:
            self.hash = hash
            self.base_xy = xy
            self.base_thresh =  self.computeEucDist(xyxy, self.wh, True)

        
            
        if self.track_state == TrackState.Confirmed: 
            if self.distance(xy, self.base_xy) > self.base_thresh:
                self.base_thresh = self.computeEucDist(xyxy, self.wh, True)
                self.hash = hash
                self.base_xy = xy
                self.is_base_changed =True
                # print(f'id >> {self.track_id} is on the move')
                
        self.hash = hash
        self.xyxy = xyxy
        self.last_seen = 0
        self.calls += 1
        self.missed = False
        self.class_id = class_id
        self.conf = conf

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.track_state == TrackState.Tentative:
            # if self.calls > self.n_init:
            # print(f'track id in init state = {self.track_id} is deleted')
            self.track_state = TrackState.Deleted
        if self.last_seen > self.max_age:
            # print(f'track id in max age = {self.track_id} is deleted')
            self.track_state = TrackState.Deleted
        if self.is_base_changed and self.last_seen > self.max_age/2:
            # print(f'deleted {self.track_id}')
            self.track_state = TrackState.Deleted
        self.last_seen += 1
        self.missed = True

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.track_state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.track_state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.track_state == TrackState.Deleted

    def is_missed(self):
        """Returns True if this track is dead and should be deleted."""
        return self.missed
