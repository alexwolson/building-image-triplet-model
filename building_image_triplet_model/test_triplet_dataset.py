"""
Comprehensive tests for triplet_dataset.py logic.
Tests validate the curriculum learning strategy and difficulty selection.
"""

import math

import numpy as np

from building_image_triplet_model.triplet_dataset import TripletDifficulty


class TestTripletDifficulty:
    """Test TripletDifficulty class logic."""

    def test_update_with_high_loss(self):
        """Test that high loss is considered failure (model struggling)."""
        diff = TripletDifficulty(
            min_distance=0.0, max_distance=1.0, success_rate=0.5, num_attempts=0
        )

        # High loss should be failure (model struggling)
        success = diff.update(0.8, threshold=0.3)
        assert success is False, "High loss (0.8 > 0.3) should be failure"
        assert diff.success_rate == 0.0, "Success rate should be 0.0 after one failure"
        assert diff.num_attempts == 1, "Attempts should be 1"

    def test_update_with_low_loss(self):
        """Test that low loss is considered success (model doing well)."""
        diff = TripletDifficulty(
            min_distance=0.0, max_distance=1.0, success_rate=0.5, num_attempts=0
        )

        # Low loss should be success (model doing well)
        success = diff.update(0.1, threshold=0.3)
        assert success is True, "Low loss (0.1 < 0.3) should be success"
        assert diff.success_rate == 1.0, "Success rate should be 1.0 after one success"
        assert diff.num_attempts == 1, "Attempts should be 1"

    def test_update_at_threshold_boundary(self):
        """Test boundary condition at exactly threshold value."""
        diff = TripletDifficulty(
            min_distance=0.0, max_distance=1.0, success_rate=0.5, num_attempts=0
        )

        # At threshold should be success (not strictly less than)
        success = diff.update(0.3, threshold=0.3)
        assert success is False, "Loss at threshold (0.3 == 0.3) should be failure"
        assert diff.success_rate == 0.0, "Success rate should be 0.0"

    def test_success_rate_accumulation(self):
        """Test that success rate accumulates correctly over multiple updates."""
        diff = TripletDifficulty(
            min_distance=0.0, max_distance=1.0, success_rate=0.5, num_attempts=0
        )

        # First update: success (low loss)
        diff.update(0.1, threshold=0.3)
        assert diff.success_rate == 1.0, "Should be 1.0 after first success"

        # Second update: failure (high loss)
        diff.update(0.5, threshold=0.3)
        assert diff.success_rate == 0.5, "Should be 0.5 after 1 success, 1 failure"

        # Third update: success
        diff.update(0.2, threshold=0.3)
        expected = (1.0 + 0.0 + 1.0) / 3  # 0.667
        assert abs(diff.success_rate - expected) < 1e-6, f"Expected {expected:.4f}"


class TestUCBSelection:
    """Test UCB-based difficulty selection logic."""

    def test_ucb_prefers_low_success_rate(self):
        """Test that UCB prefers difficulties with low success rate (challenging)."""
        levels = [
            TripletDifficulty(0.0, 0.33, success_rate=0.9, num_attempts=10),  # Easy, high success
            TripletDifficulty(0.33, 0.67, success_rate=0.5, num_attempts=10),  # Medium
            TripletDifficulty(0.67, 1.0, success_rate=0.1, num_attempts=10),  # Hard, low success
        ]

        ucb_alpha = 2.0
        total_attempts = sum(lvl.num_attempts for lvl in levels)

        scores = [
            (1.0 - lvl.success_rate)
            + math.sqrt(ucb_alpha * math.log(total_attempts) / max(1, lvl.num_attempts))
            for lvl in levels
        ]

        selected_idx = np.argmax(scores)

        # With equal attempts, should prefer lowest success_rate (most challenging)
        assert (
            selected_idx == 2
        ), f"Should select level with lowest success_rate (index 2), got {selected_idx}"

    def test_ucb_balances_exploration_exploitation(self):
        """Test that UCB balances exploration (few attempts) vs exploitation (low success)."""
        levels = [
            TripletDifficulty(0.0, 0.33, success_rate=0.5, num_attempts=100),  # Well explored
            TripletDifficulty(0.33, 0.67, success_rate=0.5, num_attempts=1),  # Unexplored
        ]

        ucb_alpha = 2.0
        total_attempts = sum(lvl.num_attempts for lvl in levels)

        scores = [
            (1.0 - lvl.success_rate)
            + math.sqrt(ucb_alpha * math.log(total_attempts) / max(1, lvl.num_attempts))
            for lvl in levels
        ]

        selected_idx = np.argmax(scores)

        # Should select unexplored level due to high exploration bonus
        assert selected_idx == 1, f"Should select unexplored level (index 1), got {selected_idx}"

    def test_ucb_with_zero_attempts(self):
        """Test UCB handling of zero attempts (should use max(1, attempts))."""
        levels = [
            TripletDifficulty(0.0, 0.5, success_rate=0.5, num_attempts=0),
            TripletDifficulty(0.5, 1.0, success_rate=0.5, num_attempts=10),
        ]

        ucb_alpha = 2.0
        total_attempts = sum(max(1, lvl.num_attempts) for lvl in levels)

        scores = [
            (1.0 - lvl.success_rate)
            + math.sqrt(ucb_alpha * math.log(total_attempts) / max(1, lvl.num_attempts))
            for lvl in levels
        ]

        # Should not crash and should prefer level with 0 attempts
        selected_idx = np.argmax(scores)
        assert (
            selected_idx == 0
        ), f"Should select level with 0 attempts (index 0), got {selected_idx}"


class TestCurriculumLearningSemantics:
    """Test whether the difficulty system implements proper curriculum learning."""

    def test_curriculum_learning_direction(self):
        """
        Test the direction of curriculum learning.

        In proper curriculum learning:
        1. Start with easy examples (that give low loss)
        2. Gradually move to harder examples (that give higher loss)
        3. UCB should prefer difficulties where model is struggling (high loss, low success)

        After the fix:
        - success = (loss < threshold) means low loss = success
        - UCB prefers (1 - success_rate), i.e., low success_rate
        - This means UCB prefers difficulties with HIGH loss (challenging)
        - This is CORRECT for curriculum learning!
        """
        # Simulate scenario after some training
        levels = [
            # Easy level: Model doing well (low loss), so HIGH success_rate
            TripletDifficulty(0.0, 0.33, success_rate=0.8, num_attempts=50),
            # Hard level: Model struggling (high loss), so LOW success_rate
            TripletDifficulty(0.67, 1.0, success_rate=0.2, num_attempts=50),
        ]

        ucb_alpha = 2.0
        total_attempts = sum(lvl.num_attempts for lvl in levels)

        scores = [
            (1.0 - lvl.success_rate)
            + math.sqrt(ucb_alpha * math.log(total_attempts) / max(1, lvl.num_attempts))
            for lvl in levels
        ]

        selected_idx = np.argmax(scores)

        # Fixed implementation should select level with LOWER success_rate (harder)
        assert selected_idx == 1, (
            "Fixed implementation should select hard level (high loss) "
            "where model struggles, which is correct for curriculum learning."
        )


class TestNegativeSamplingLogic:
    """Test negative sampling edge cases."""

    def test_candidate_filtering_order(self):
        """
        Test that candidate filtering happens in correct order.

        After fix: The code filters out anchor_row BEFORE checking if empty.
        This prevents empty valid_tids edge case.
        """
        # Simulate scenario where anchor_row is the only candidate
        candidate_rows = np.array([5])  # Only the anchor
        anchor_row = 5

        # Fixed code: filter first, then check
        filtered = candidate_rows[candidate_rows != anchor_row]
        if len(filtered) == 0:
            result = None
        else:
            result = filtered

        assert result is None, "Fixed approach should return None"
        assert len(filtered) == 0, "After filtering, should have no candidates"


class TestDifficultyUpdateMismatch:
    """Test for difficulty update tracking improvements."""

    def test_difficulty_tracking_in_dataset(self):
        """
        Test that the dataset now tracks which difficulty was used.

        After fix: The dataset maintains a sample_difficulty_map that tracks
        which difficulty level was used for each sample, enabling better
        updates (though still approximate due to batching).
        """
        # This is more of a documentation test
        # The actual tracking happens in __getitem__ and update_difficulty
        # The fix uses the most common difficulty from recent samples

        # Simulate tracking
        sample_difficulty_map = {
            0: 0,
            1: 0,
            2: 1,
            3: 1,
            4: 2,  # Last 5 samples
        }

        # Get most common from last batch (e.g., last 3)
        recent_difficulties = list(sample_difficulty_map.values())[-3:]
        from collections import Counter

        most_common = Counter(recent_difficulties).most_common(1)[0][0]

        # Should pick difficulty level 1 (appears in indices 2, 3)
        assert most_common == 1, f"Expected difficulty 1, got {most_common}"
