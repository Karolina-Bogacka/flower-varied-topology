from abc import ABC
from typing import List

from flwr.common import Parameters, NDArrays


class Hypernetwork(ABC):
    """Abstract base class for hypernetworks used in peFedHN strategy."""

    def fit(self, weights_ins: NDArrays, weights_res: NDArrays):
        """Train the model in order to produce the best personalized weights for a given
        model based on local weight updates.

        Parameters
        ----------
        weights_ins : NDArrays
            The training weights assigned to a given client network beforehand.
        weights_res : NDArrays
            The training weights obtained from the client as a result of local
            training.
        """

    def predict(self, weights_ins: NDArrays) -> NDArrays:
        """Predict the best weights for a given client based on its current weights.

        Parameters
        ----------
        weights_ins : NDArrays
            The training weights serving as a basis for further prediction.

        Output
        ----------
        NDArrays
            The training weights generated by the hypernetwork.
        """
