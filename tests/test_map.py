import pandas as pd
import pytest

from tests.base_test import BaseTest


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Course Name": [
                "Introduction to Programming",
                "Advanced Programming",
                "Cooking Basics",
                "Advanced Culinary Arts",
                "Data Structures",
                "Algorithms",
                "French Cuisine",
                "Italian Cooking",
            ],
            "Department": ["CS", "CS", "Culinary", "Culinary", "CS", "CS", "Culinary", "Culinary"],
            "Level": [100, 200, 100, 200, 300, 300, 200, 200],
        }
    )


class TestMap(BaseTest):
    def test_sem_map_has_provenance(self, sample_df):
        """Test sem_map functionality with provenance"""
        # choose arbitrary indices
        sample_df.index = [101, 202, 303, 404, 505, 606, 707, 808]

        shuffled_df = sample_df.sample(frac=1)
        user_instruction = "Summarize {Course Name} in one word only."
        result = shuffled_df.sem_map(user_instruction, return_provenance=True)

        assert "provenance_id_map" in result.columns

        for idx, row in result.iterrows():
            prov_id = row["provenance_id_map"]
            assert prov_id == idx

            # verify data matches that provenance id will point to correct data
            original_course = sample_df.loc[prov_id, "Course Name"]
            assert row["Course Name"] == original_course

    def test_sem_map_provenance_empty_result(self):
        """Test for correctness of empty DataFrame for sem_map with provenance"""
        empty_df = pd.DataFrame(columns=["Course Name", "Department", "Level"])
        user_instruction = "What is a similar course to {Course Name} "
        result = empty_df.sem_map(user_instruction, return_provenance=True)

        assert len(result) == 0
        assert "provenance_id_map" in result.columns
