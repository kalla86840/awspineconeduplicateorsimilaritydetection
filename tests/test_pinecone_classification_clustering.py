import unittest

from rag_endpoint.app import cluster_vector_matches, pinecone_matches_with_values


class PineconeClassificationClusteringTests(unittest.TestCase):
    def test_pinecone_matches_with_values_preserves_vectors(self):
        result = {
            "matches": [
                {
                    "id": "record-1",
                    "score": 0.91,
                    "values": [0.1, 0.2],
                    "metadata": {"title": "Edge cloud", "category": "cloud"},
                }
            ]
        }

        matches = pinecone_matches_with_values(result)

        self.assertEqual(matches[0]["id"], "record-1")
        self.assertEqual(matches[0]["values"], [0.1, 0.2])
        self.assertEqual(matches[0]["metadata"]["category"], "cloud")

    def test_cluster_vector_matches_groups_nearby_vectors(self):
        matches = [
            {"id": "a", "title": "A", "score": 0.9, "metadata": {}, "values": [0.0, 0.0]},
            {"id": "b", "title": "B", "score": 0.8, "metadata": {}, "values": [0.1, 0.0]},
            {"id": "c", "title": "C", "score": 0.7, "metadata": {}, "values": [9.0, 9.0]},
            {"id": "d", "title": "D", "score": 0.6, "metadata": {}, "values": [9.1, 9.0]},
        ]

        clusters = cluster_vector_matches(matches, 2)
        sizes = sorted(cluster["size"] for cluster in clusters)

        self.assertEqual(sizes, [2, 2])


if __name__ == "__main__":
    unittest.main()
