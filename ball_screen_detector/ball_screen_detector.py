import sys
sys.path.append('../')
from utils.bbox_utils import get_center_of_bbox, measure_distance
from collections import defaultdict


class BallScreenDetector:
    """
    Detects ball screen (pick) events by combining player tracking,
    team assignment, and ball possession data.

    Pipeline:
      1. Geometric trigger  — same team, one has ball, bbox distance < threshold
      2. Sliding window     — track candidate pair across frames
      3. Feature extraction — velocity drop, distance stability, relative position
      4. Rule classifier    — confirm screen or reject
      5. Temporal smoothing — require N consecutive confirmed frames
    """

    def __init__(self,
                 proximity_threshold=120,
                 min_screen_frames=8,
                 screener_stillness_window=6,
                 stillness_velocity_threshold=8.0):
        """
        Args:
            proximity_threshold (float): Max pixel distance between bbox centers
                to consider a pair as a screen candidate.
            min_screen_frames (int): Consecutive frames needed to confirm a screen.
            screener_stillness_window (int): Number of recent frames to measure
                screener velocity for stillness check.
            stillness_velocity_threshold (float): Max avg pixel movement per frame
                for a player to be considered "set" (stationary).
        """
        self.proximity_threshold = proximity_threshold
        self.min_screen_frames = min_screen_frames
        self.screener_stillness_window = screener_stillness_window
        self.stillness_velocity_threshold = stillness_velocity_threshold

        # {(ball_handler_id, candidate_id): [center_positions_of_candidate, ...]}
        self._candidate_history = defaultdict(list)
        self._consecutive_screen_count = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_screens(self, player_tracks, player_assignment, ball_aquisition):
        """
        Run screen detection over all frames.

        Args:
            player_tracks (list[dict]): [{player_id: {"bbox": [x1,y1,x2,y2]}}]
            player_assignment (list[dict]): [{player_id: team_id}]
            ball_aquisition (list[int]): [player_id or -1] per frame

        Returns:
            list[dict or None]: Per-frame result.
                None  → no screen detected this frame
                dict  → {"screener_id": int, "ball_handler_id": int, "team": int,
                          "screener_bbox": list, "ball_handler_bbox": list}
        """
        num_frames = len(player_tracks)
        screen_events = [None] * num_frames

        for frame_num in range(num_frames):
            ball_handler_id = ball_aquisition[frame_num]
            frame_tracks = player_tracks[frame_num]
            frame_teams = player_assignment[frame_num]

            if ball_handler_id == -1 or ball_handler_id not in frame_tracks:
                self._decay_all_candidates()
                continue

            handler_bbox = frame_tracks[ball_handler_id]["bbox"]
            handler_team = frame_teams.get(ball_handler_id)
            handler_center = get_center_of_bbox(handler_bbox)

            # Find same-team players within proximity threshold
            candidate_ids = self._find_candidates(
                ball_handler_id, handler_center, handler_team,
                frame_tracks, frame_teams
            )

            # Update history for each candidate pair
            active_pairs = set()
            for candidate_id in candidate_ids:
                pair_key = (ball_handler_id, candidate_id)
                active_pairs.add(pair_key)

                candidate_bbox = frame_tracks[candidate_id]["bbox"]
                candidate_center = get_center_of_bbox(candidate_bbox)
                self._candidate_history[pair_key].append(candidate_center)

                # Keep only the last window we need
                max_history = self.screener_stillness_window + 2
                if len(self._candidate_history[pair_key]) > max_history:
                    self._candidate_history[pair_key].pop(0)

                # Classify this pair
                if self._is_screen(pair_key, handler_center):
                    self._consecutive_screen_count[pair_key] += 1
                else:
                    self._consecutive_screen_count[pair_key] = 0

                if self._consecutive_screen_count[pair_key] >= self.min_screen_frames:
                    screen_events[frame_num] = {
                        "screener_id": candidate_id,
                        "ball_handler_id": ball_handler_id,
                        "team": handler_team,
                        "screener_bbox": candidate_bbox,
                        "ball_handler_bbox": handler_bbox,
                    }
                    break  # Report only the best (closest) pair per frame

            # Decay pairs that are no longer close
            self._decay_inactive_pairs(active_pairs)

        return screen_events

    # ------------------------------------------------------------------
    # Stage 1 — Geometric Trigger
    # ------------------------------------------------------------------

    def _find_candidates(self, ball_handler_id, handler_center,
                         handler_team, frame_tracks, frame_teams):
        """
        Return same-team player IDs sorted by distance to ball handler,
        filtered by proximity threshold.
        """
        candidates = []
        for player_id, info in frame_tracks.items():
            if player_id == ball_handler_id:
                continue
            if frame_teams.get(player_id) != handler_team:
                continue  # Different team — cannot be screener

            candidate_center = get_center_of_bbox(info["bbox"])
            dist = measure_distance(handler_center, candidate_center)
            if dist < self.proximity_threshold:
                candidates.append((player_id, dist))

        # Closest first — most likely screener
        candidates.sort(key=lambda x: x[1])
        return [pid for pid, _ in candidates]

    # ------------------------------------------------------------------
    # Stage 2 — Feature-based Classifier (rule-based, swappable)
    # ------------------------------------------------------------------

    def _is_screen(self, pair_key, handler_center):
        """
        Decide if the candidate is currently acting as a screener.

        Rules:
          1. We have enough history to judge stillness.
          2. Screener's average velocity over recent frames is below threshold
             (player is "set" — not still running).

        This method can be replaced with a LightGBM predict call once
        a model is trained. The features extracted here become the model input.
        """
        history = self._candidate_history[pair_key]
        if len(history) < self.screener_stillness_window:
            return False  # Not enough history yet

        features = self._extract_features(history, handler_center)
        return self._rule_classifier(features)

    def _extract_features(self, candidate_history, handler_center):
        """
        Extract numeric features from the candidate's position history.
        These same features can be fed into LightGBM as a drop-in replacement.

        Returns:
            dict of float features
        """
        window = candidate_history[-self.screener_stillness_window:]

        # Velocity: frame-to-frame displacement of candidate center
        velocities = [
            measure_distance(window[i], window[i - 1])
            for i in range(1, len(window))
        ]

        avg_velocity = sum(velocities) / len(velocities) if velocities else 0.0
        last3_velocity = sum(velocities[-3:]) / 3 if len(velocities) >= 3 else avg_velocity

        # Velocity drop: was moving before, now stationary?
        early_velocity = sum(velocities[:3]) / 3 if len(velocities) >= 3 else avg_velocity
        velocity_drop = early_velocity - last3_velocity  # Positive = slowing down

        # Distance from current candidate position to ball handler
        current_pos = candidate_history[-1]
        dist_to_handler = measure_distance(current_pos, handler_center)

        # Distance trend: getting closer (negative) or moving away (positive)
        if len(candidate_history) >= 4:
            old_dist = measure_distance(candidate_history[-4], handler_center)
            dist_trend = dist_to_handler - old_dist
        else:
            dist_trend = 0.0

        return {
            "avg_velocity": avg_velocity,
            "last3_velocity": last3_velocity,
            "velocity_drop": velocity_drop,
            "dist_to_handler": dist_to_handler,
            "dist_trend": dist_trend,
        }

    def _rule_classifier(self, features):
        """
        Rule-based classification using extracted features.

        Replace this method body with:
            return self.lgbm_model.predict([list(features.values())])[0] > 0.5
        once LightGBM model is trained.
        """
        # Rule 1: Screener must be nearly stationary in recent frames
        is_still = features["last3_velocity"] < self.stillness_velocity_threshold

        # Rule 2: Screener must have slowed down (came to set position)
        is_setting = features["velocity_drop"] > 0

        # Rule 3: Must remain close to ball handler (not drifting away)
        is_close = features["dist_to_handler"] < self.proximity_threshold
        not_leaving = features["dist_trend"] < 15  # Not moving away fast

        return is_still and is_setting and is_close and not_leaving

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _decay_all_candidates(self):
        self._candidate_history.clear()
        self._consecutive_screen_count.clear()

    def _decay_inactive_pairs(self, active_pairs):
        """Reset counter for pairs no longer in proximity."""
        stale = [k for k in self._consecutive_screen_count if k not in active_pairs]
        for k in stale:
            self._consecutive_screen_count.pop(k, None)
            self._candidate_history.pop(k, None)
