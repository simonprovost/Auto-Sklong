from typing import Optional

from gama.genetic_programming.components import Individual, PrimitiveNode
from gama.genetic_programming.mutation import has_multiple_options
import ConfigSpace as cs

from gama.utilities.config_space import get_internal_longitudinal_output_types


class LongitudinalSearchSpaceConstraint:
    """LongitudinalSearchSpaceConstraint manage and apply constraints for
    genetic programming in a longitudinal context.

    This class handles the exclusion of specific preprocessors and estimators based on
    the data preparation method used. It allows for dynamic determination of valid
    mutations and crossovers within the genetic programming process.

    The constraints dict should be of the following format:
    {
        "DataPrepMethod1": {
            "preprocessors": ["Preprocessor1", "Preprocessor2", ...],
            "estimators": ["Estimator1", "Estimator2", ...]
        },
        "DataPrepMethod2": {
            "preprocessors": ["Preprocessor3", "Preprocessor4", ...],
            "estimators": ["Estimator3", "Estimator4", ...]
        },
        ...
        "Default": {
            "preprocessors": ["DefaultPreprocessor1", ...],
            "estimators": ["DefaultEstimator1", ...]
        }
    }

    How to read this format is as follows:
    - Each top-level key (e.g., "DataPrepMethod1") represents a specific data
    preparation method.
    - Under each data preparation method, there are keys for "preprocessors"
    and "estimators".
    - Each of these keys maps to a list of names of preprocessors or estimators that are
      excluded from being selected when the respective data preparation method is used.
    - The "Default" key provides fallback exclusions for any data preparation method
    not explicitly listed.

    Attributes
    __________
        constraints (dict): A dictionary defining the constraints for each data
        preparation method.

    Raises
    ______
        ValueError: If the provided constraints dictionary is invalid or missing
        required keys.
    """

    def __init__(self, constraints: dict):
        if not constraints or not isinstance(constraints, dict):
            raise ValueError(
                "No constraints provided."
                "Please provide valid constraints under a Dict format."
            )

        required_keys = {"preprocessors", "estimators"}
        for data_prep_method, constraint in constraints.items():
            if not isinstance(constraint, dict) or required_keys - constraint.keys():
                missing_keys = required_keys - constraint.keys()
                missing_keys_str = ", ".join(missing_keys)
                raise ValueError(
                    f"Constraint for '{data_prep_method}' "
                    f"must be a dictionary with keys {required_keys}. "
                    f"Missing keys: {missing_keys_str}"
                )

        self.constraints = constraints

    def get_exclusions_for_node(self, individual, node_type):
        """Determines the list of excluded options for a given node type

        Focus: of an individual

        Args
        ____
            individual (Individual): The individual being mutated.
            node_type (str): The type of node for which to determine exclusions.
            Should be one of 'data_preparation', 'preprocessors', or 'estimators'.

        Returns
        _______
            list: A list of strings representing the excluded options for the specified
            node type.
        """
        data_prep_str = self.find_data_preparation_node(individual)._primitive.__str__()

        constraints_for_method = self.constraints.get(
            data_prep_str, self.constraints.get("Default", {})
        )
        return constraints_for_method.get(node_type, [])

    def can_crossover(self, ind1_primitive, ind2_primitive):
        """Determines if two individuals can crossover

        Focus: based on their data preparation methods.

        One specific case is when the data preparation method is MerWavTimePlus, in
        which case the two individuals can only crossover if they have the same data
        preparation method.

        Args
        ____
            ind1_primitive (PrimitiveNode): The data preparation primitive of the
            first individual.
            ind2_primitive (PrimitiveNode): The data preparation primitive of the
            second individual.

        Returns
        _______
            bool: True if crossover is allowed between the two individuals,
            False otherwise.
        """
        ind1_data_preparation_primitive = self.find_data_preparation_node(
            ind1_primitive
        )._primitive
        ind2_data_preparation_primitive = self.find_data_preparation_node(
            ind2_primitive
        )._primitive

        if "MerWavTimePlus" in [
            ind1_data_preparation_primitive.__str__(),
            ind2_data_preparation_primitive.__str__(),
        ]:
            return ind1_data_preparation_primitive == ind2_data_preparation_primitive
        else:
            return True

    @staticmethod
    def find_data_preparation_node(individual: Individual):
        """Finds the data preparation node in a given individual.

        Traverses the structure of an individual to locate the node responsible for
        data preparation.

        Args
        ____
            individual (Individual): The individual to search within.

        Returns
        _______
            PrimitiveNode: The data preparation node found in the individual.

        Raises
        ______
            ValueError: If the individual does not have a data preparation node.
        """
        data_preparation_node = individual.main_node
        while data_preparation_node._primitive.output != "data_preparation":
            data_preparation_node = data_preparation_node._data_node  # type: ignore
            if not isinstance(data_preparation_node, PrimitiveNode):
                raise ValueError("Individual has no data_preparation node")
        return data_preparation_node

    @staticmethod
    def has_longitudinal_multiple_options(
        hyperparameter: cs.hyperparameters.hyperparameter,
        optional_hyperparameter_choice_name: Optional[str] = None,
    ) -> bool:
        """Checks if a ConfigSpace hyperparameter has more than one longitudinal option.

        One specific case is when the data preparation method is MerWavTimePlus, in
        which case the hyperparameter has only one option, so no multiple options.

        Args
        ____
            hyperparameter (cs.hyperparameters.hyperparameter): The hyperparameter to
            check.
            optional_hyperparameter_choice_name (str, optional): An optional
            hyperparameter choice name to consider. Defaults to None.

        Returns
        _______
            bool: True if the hyperparameter has more than one longitudinal option,
            False otherwise.
        """
        if isinstance(hyperparameter, cs.CategoricalHyperparameter):
            if "MerWavTimePlus" in hyperparameter.choices:
                if (
                    isinstance(optional_hyperparameter_choice_name, str)
                    and "MerWavTimePlus" in optional_hyperparameter_choice_name
                ):
                    return False
                return len(set(hyperparameter.choices)) > 2
            return len(set(hyperparameter.choices)) > 1
        return has_multiple_options(hyperparameter)

    @staticmethod
    def primitive_replaceable(index_primitive, config_space):
        """Determines if a primitive in an individual is replaceable

        Focus: based on longitudinal options

        One specific case is when the primitive output is a preprocessor, in which case
        the primitive is not replaceable given that the search space for preprocessors
        is currently limited to one option per preprocessor.

        Args
        ____
            index_primitive (tuple): A tuple containing the index and the
            primitive node.
            config_space (cs.ConfigurationSpace): The configuration space to
            consider for replacement options.

        Returns
        _______
            bool: True if the primitive is replaceable, False otherwise.
        """
        _, primitive = index_primitive
        if primitive._primitive.output == "preprocessors":
            return False
        return LongitudinalSearchSpaceConstraint.has_longitudinal_multiple_options(
            config_space.get_hyperparameter(
                config_space.meta[primitive._primitive.output]
                if primitive._primitive.output
                in get_internal_longitudinal_output_types()
                else primitive._primitive.output
            ),
            primitive.__str__(),
        )
