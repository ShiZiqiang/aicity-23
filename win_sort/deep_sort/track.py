# vim: expandtab:ts=4:sw=4
import numpy as np
from . import kalman_filter
import strong_sort.opt as opt

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, detection, track_id, n_init, max_age,
                 feature, score, classification):
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            feature /= np.linalg.norm(feature)
            self.features.append(feature)

        self.scores = []
        if score is not None:
            self.scores.append(score)

        self._n_init = n_init
        self._max_age = max_age
        self.classification = classification

        #self.kf = kalman_filter.KalmanFilter()
        # twin kalman filter
        self.pkf = kalman_filter.PKalmanFilter()
        self.bkf = kalman_filter.BKalmanFilter()

        #self.mean, self.covariance = self.kf.initiate(detection)
        #print('detection {}'.format(detection))
        self.mean_p, self.covariance_p = \
            self.pkf.initiate([detection[0],detection[1]], detection[3])
        self.mean_b, self.covariance_b = \
            self.bkf.initiate([detection[2],detection[3]])

        #self.mean = [self.mean_p,  self.mean_b]
        self.mean = np.array([self.mean_p[0], self.mean_p[1], self.mean_b[0],  self.mean_b[1],
                     self.mean_p[2], self.mean_p[3], self.mean_b[2],  self.mean_b[3]])

        # print('self.mean_p {}'.format(self.mean_p))
        # print('self.mean_b {}'.format(self.mean_b))
        #print('self.mean[:4] {}'.format(self.mean[:4]))

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        #print('ret {}'.format(ret))
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        """
        #self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        #print('self.mean {}'.format(self.mean))

        self.mean_p, self.covariance_p = \
            self.pkf.predict(self.mean_p, self.covariance_p, self.mean[3])
        self.mean_b, self.covariance_b = \
            self.bkf.predict(self.mean_b, self.covariance_b)

        self.mean =  np.array([self.mean_p[0], self.mean_p[1], self.mean_b[0], self.mean_b[1],
                     self.mean_p[2], self.mean_p[3], self.mean_b[2], self.mean_b[3]])

        self.age += 1
        self.time_since_update += 1

    @staticmethod
    def get_matrix(dict_frame_matrix, frame):
        eye = np.eye(3)
        matrix = dict_frame_matrix[frame]
        dist = np.linalg.norm(eye - matrix)
        if dist < 100:
            return matrix
        else:
            return eye

    def update(self, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        detection : Detection
            The associated detection.

        """
        # self.mean, self.covariance = self.kf.update(self.mean, self.covariance, detection.to_xyah(), detection.confidence)

        self.mean_p, self.covariance_p = \
            self.pkf.update(self.mean_p, self.covariance_p, detection.to_xyah()[0:2],
                                                                self.mean_b[1], detection.confidence)
        self.mean_b, self.covariance_b = \
            self.bkf.update(self.mean_b, self.covariance_b, detection.to_xyah()[2:4],
                                                    detection.confidence)

        self.mean =  np.array([self.mean_p[0], self.mean_p[1], self.mean_b[0], self.mean_b[1],
                     self.mean_p[2], self.mean_p[3], self.mean_b[2], self.mean_b[3]])

        feature = detection.feature / np.linalg.norm(detection.feature)
        if opt.EMA:
            smooth_feat = opt.EMA_alpha * self.features[-1] + (1 - opt.EMA_alpha) * feature
            smooth_feat /= np.linalg.norm(smooth_feat)
            self.features = [smooth_feat]
        else:
            self.features.append(feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        
        self.classification = detection.classification

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
