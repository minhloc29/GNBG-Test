import json
import uuid
import numpy as np


class Solution:
    """
    Represents a candidate solution (an individual) in the evolutionary algorithm.
    Each individual has properties such as code, fitness, feedback, and metadata for additional information.
    """

    def __init__(
        self,
        code="",
        name="",
        description="",
        configspace=None,
        generation=0,
        parent_ids=[],
        operator=None,
        role_prompt_index = 0
    ):
        """
        Initializes an individual with optional attributes.

        Args:
            code (str): The code of the individual.
            name (str): The name of the individual (typically the class name in the code).
            description (str): A short description of the individual (e.g., algorithm's purpose or behavior).
            configspace (Optional[ConfigSpace]): Optional configuration space for HPO.
            generation (int): The generation this individual belongs to.
            parent_ids (list): UUID of the parent individuals in a list.
            operator (str): Optional identifier of the LLM operation that created this individual.
        """
        self.id = str(uuid.uuid4())  # Unique ID for this individual
        self.code = code
        self.name = name
        self.description = description
        self.configspace = configspace
        self.generation = generation
        self.fitness = -np.Inf
        self.aocc_1 = -np.Inf
        self.aocc_2 = -np.Inf
        self.aocc_3 = -np.Inf
        self.feedback = ""
        self.error = ""
        self.parent_ids = parent_ids
        self.metadata = {}  # Dictionary to store additional metadata
        self.operator = operator
        self.role_prompt_index = role_prompt_index
        self.try_hs = False
    def set_operator(self, operator):
        """
        Sets the operator name that generated this individual.

        Args:
            operator (str): The name of the operator (for logging purposes).
        """
        self.operator = operator

    def add_metadata(self, key, value):
        """
        Adds key-value pairs to the metadata dictionary.

        Args:
            key (str): The key for the metadata.
            value: The value associated with the key.
        """
        self.metadata[key] = value

    def get_metadata(self, key):
        """
        Get a metadata item from the dictionary.

        Args:
            key (str): The key for the metadata to obtain.
        """
        return self.metadata[key] if key in self.metadata.keys() else None

    def set_scores(self, fitness, feedback="", error="", aocc1 = None, aocc2 = None, aocc3 = None):
        self.fitness = fitness
        self.aocc1 = aocc1
        self.aocc2 = aocc2
        self.aocc3 = aocc3
        self.feedback = feedback
        self.error = error

    def get_summary(self):
        """
        Returns a string summary of this solution's key attributes.

        Returns:
            str: A string representing the solution in a summary format.
        """
        return f"{self.name}: {self.description} (Score: {self.fitness}), RoleIdx: {self.role_prompt_index}"

    def copy(self):
        """
        Returns a copy of this solution, with a new unique ID and a reference to the current solution as its parent.

        Returns:
            Individual: A new instance of Individual with the same attributes but a different ID.
        """
        new_solution = Solution(
            code=self.code,
            name=self.name,
            description=self.description,
            configspace=self.configspace,
            generation=self.generation + 1,
            parent_ids=[self.id],  # Link this solution as the parent
            operator=self.operator,
            role_prompt_index=self.role_prompt_index
        )
        new_solution.metadata = self.metadata.copy()  # Copy the metadata as well
        return new_solution

    def to_dict(self):
        """
        Converts the individual to a dictionary.

        Returns:
            dict: A dictionary representation of the individual.
        """
        try:
            cs = self.configspace
            cs = cs.to_serialized_dict()
        except Exception as e:
            cs = ""
        return {
            "id": self.id,
            "fitness": self.fitness,
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "configspace": cs,
            "generation": self.generation,
            "feedback": self.feedback,
            "error": self.error,
            "parent_ids": self.parent_ids,
            "operator": self.operator,
            "metadata": self.metadata,
        }

    def to_json(self):
        """
        Converts the individual to a JSON string.

        Returns:
            str: A JSON string representation of the individual.
        """
        return json.dumps(self.to_dict(), default=str, indent=4)
