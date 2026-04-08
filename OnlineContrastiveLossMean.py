from enum import Enum
from torch import nn, Tensor
from typing import Iterable
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss
    
    EUCLIDEAN: Euclidean distance between vectors
    MANHATTAN: Manhattan distance between vectors
    COSINE_DISTANCE: Cosine similarity between vectors
    """

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)

class OnlineContrastiveLossMean(nn.Module):
    """
    Modification of the OnlineContrastiveLoss class.
    """
    def __init__(
        self, model: SentenceTransformer, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.5
    ) -> None:
        """
        Initialize the OnlineContrastiveLossMean module.
        
        :param model: SentenceTransformer model
        :param distance_metric: The distance metric to use. Default is cosine similarity.
        :param margin: The margin to use. Default is 0.5.
        """
        super().__init__()
        self.model = model
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor, size_average=False) -> Tensor:
        """
        Forward pass of the OnlineContrastiveLossMean module.
        
        :param sentence_features: A list of dictionaries containing the sentence features.
        :param labels: A tensor containing the labels.
        :return: A tensor containing the loss.
        """
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        # compute the distance matrix
        distance_matrix = self.distance_metric(embeddings[0], embeddings[1])
        
        # select hard positive and hard negative pairs
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]
        
        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]
        
        # compute the loss
        positive_loss = positive_pairs.pow(2).mean()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).mean()
        loss = positive_loss + negative_loss
        return loss
