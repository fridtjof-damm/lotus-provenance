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


# @pytest.mark.skip(reason="sem_index issue to be resolved")
class TestSearch(BaseTest):
    def test_basic_search(self, sample_df):
        """Test basic semantic search functionality"""
        df = sample_df.sem_index("Course Name", "course_index")
        result = df.sem_search("Course Name", "programming courses", K=2)
        assert len(result) == 2
        assert "Introduction to Programming" in result["Course Name"].values
        assert "Advanced Programming" in result["Course Name"].values

    def test_filtered_search_relational(self, sample_df):
        """Test semantic search with relational filter"""
        # Index the dataframe
        df = sample_df.sem_index("Course Name", "course_index")

        # Apply relational filter and search
        filtered_df = df[df["Department"] == "CS"]
        result = filtered_df.sem_search("Course Name", "advanced courses", K=2)

        assert len(result) == 2
        # Should only return CS courses
        assert all(dept == "CS" for dept in result["Department"])
        assert "Advanced Programming" in result["Course Name"].values

    def test_filtered_search_semantic(self, sample_df):
        """Test semantic search after semantic filter"""
        # Index the dataframe
        df = sample_df.sem_index("Course Name", "course_index")

        # Apply semantic filter and search
        filtered_df = df.sem_filter("{Course Name} is related to cooking")
        result = filtered_df.sem_search("Course Name", "advanced level courses", K=2)

        assert len(result) == 2
        # Should only return cooking-related courses
        assert all(dept == "Culinary" for dept in result["Department"])
        assert "Advanced Culinary Arts" in result["Course Name"].values

    def test_filtered_search_combined(self, sample_df):
        """Test semantic search with both relational and semantic filters"""
        # Index the dataframe
        df = sample_df.sem_index("Course Name", "course_index")

        # Apply both filters and search
        filtered_df = df[df["Level"] >= 200]  # relational filter
        filtered_df = filtered_df.sem_filter("{Course Name} is related to computer science")  # semantic filter
        result = filtered_df.sem_search("Course Name", "data structures and algorithms", K=2)

        assert len(result) == 2
        # Should only return advanced CS courses
        assert all(dept == "CS" for dept in result["Department"])
        assert all(level >= 200 for level in result["Level"])
        assert "Data Structures" in result["Course Name"].values
        assert "Algorithms" in result["Course Name"].values

    @pytest.mark.skip(reason="k-NN = 0 edge case not yet handled")
    def test_filtered_search_empty_result(self, sample_df):
        """Test semantic search when filter returns empty result"""
        df = sample_df.sem_index("Course Name", "course_index")

        # Apply filter that should return no results
        filtered_df = df[df["Level"] > 1000]
        result = filtered_df.sem_search("Course Name", "any course", K=2)

        assert len(result) == 0

    def test_filtered_search_with_scores(self, sample_df):
        """Test filtered semantic search with similarity scores"""
        df = sample_df.sem_index("Course Name", "course_index")

        filtered_df = df[df["Department"] == "CS"]
        result = filtered_df.sem_search("Course Name", "programming courses", K=2, return_scores=True)

        assert "vec_scores_sim_score" in result.columns
        assert len(result["vec_scores_sim_score"]) == 2
        # Scores should be between 0 and 1
        assert all(0 <= score <= 1 for score in result["vec_scores_sim_score"])

    def test_filter_correct_provenance_id(self, sample_df):
        """
        Test that provenance_id correctly tracks the original tuple index when using sem_filter.
        """
        # Prepare filtered dataframe and original indices
        filtered_df = sample_df[sample_df["Department"] == "CS"]
        original_indices = set(filtered_df.index.tolist())

        result = filtered_df.sem_filter("{Course Name} is related to programming", return_provenance=True)

        # Check that provenance_id column exists and ids are part of original indices
        assert "provenance_id" in result.columns
        prov_ids = list(result["provenance_id"])
        assert set(prov_ids).issubset(original_indices)

        # each provenance_id should map to the correct original row
        for _, row in result.iterrows():
            pid = row["provenance_id"]
            assert sample_df.loc[pid, "Course Name"] == row["Course Name"]
            assert sample_df.loc[pid, "Department"] == row["Department"]
            assert sample_df.loc[pid, "Level"] == row["Level"]

    def test_empty_filter_result_with_provenance(self, sample_df):
        """
        Test that sem_filter returns an empty DataFrame with provenance_id column
        when no rows match the filter criteria.
        """
        filtered_df = sample_df[sample_df["Level"] > 1000]  # No such level exists

        result = filtered_df.sem_filter("{Course Name} is related to programming", return_provenance=True)

        assert "provenance_id" in result.columns
        assert len(result) == 0
