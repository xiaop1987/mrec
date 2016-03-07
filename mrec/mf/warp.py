import numpy as np
import random

from mrec.evaluation import metrics

from recommender import MatrixFactorizationRecommender
from model.warp import WARP, WARPLimitedItem


class WARPMFRecommender(MatrixFactorizationRecommender):
    """
    Learn matrix factorization optimizing the WARP loss.

    Parameters
    ==========
    d : int
        Dimensionality of factors.
    gamma : float
        Learning rate.
    C : float
        Regularization constant.
    batch_size : int
        Mini batch size for SGD updates.
    positive_thresh: float
        Consider an item to be "positive" i.e. liked if its rating is at least this.
    max_iters : int
        Number of attempts allowed to find a violating negative example during updates.
        In practice it means that we optimize for ranks 1 to max_iters-1.
    """

    def __init__(self,d,gamma,C,batch_size=10,positive_thresh=0.00001,max_iters=1000,
                 validation_iters = 100, max_trials = 50, sample_item_rate = 1.0,
                 model_type = 'WARP'):
        self.d = d
        self.gamma = gamma
        self.C = C
        self.batch_size = batch_size
        self.positive_thresh = positive_thresh
        self.max_iters = max_iters
        self.validation_iters = validation_iters
        self.max_trials = max_trials
        self.sample_item_rate = sample_item_rate
        self.model_type = model_type
        self.model = None
        if self.model_type == 'WARP':
            self.model = WARP(self.d, self.gamma, self.C, self.max_iters,
                              self.validation_iters,self.batch_size, self.positive_thresh,
                              self.max_trials)
        elif self.model_type == 'WARPLimitedItem':
            self.model = WARPLimitedItem(self.d, self.gamma, self.C, self.max_iters,
                                         self.validation_iters,self.batch_size, self.positive_thresh,
                                         self.max_trials, self.sample_item_rate)
        else:
            print "Invalid model type:[%s]" % self.model_type
            sys.exit(2)

    def recommend(self, users, topK):
        return self.model.recommend(users, topK)

    def fit(self,train,item_features=None):
        """
        Learn factors from training set.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix
            User-item matrix.
        item_features : array_like, shape = [num_items, num_features]
            Features for each item in the dataset, ignored here.
        """
        max_iters,validation_iters,validation = self.create_validation_set(train)
        self.description = 'WARPMF({0})'.format(self.model)
        print "Start model fit with param:[%s]" % self.description
        self.model.fit(train, validation)

        self.U = self.model.U_
        self.V = self.model.V_


    @staticmethod
    def create_validation_set(train):
        """
        Hide and return half of the known items for a sample of users,
        and estimate the number of sgd iterations to run.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix
            User-item matrix.

        Returns
        =======
        max_iters : int
            Total number of sgd iterations to run.
        validation_iters : int
            Check progress after this many iterations.
        validation : dict
            Validation set.
        """
        # use 1% of users for validation, with a floor
        num_users = train.shape[0]
        num_validation_users = max(num_users/100,40)
        # ensure reasonable expected number of updates per validation user
        validation_iters = 100*num_users/num_validation_users
        # and reasonable number of validation cycles
        max_iters = 30*validation_iters

        print num_validation_users,'validation users'
        print validation_iters,'validation iters'
        print max_iters,'max_iters'

        validation = dict()
        for i in xrange(num_validation_users):
            u = random.randint(0, num_users - 1)
            positive = np.where(train[u].data > 0)[0]
            hidden = random.sample(positive,positive.shape[0]/2)
            if hidden:
                train[u].data[hidden] = 0
                validation[u] = train[u].indices[hidden]

        return max_iters,validation_iters,validation

def main():
    import sys
    from mrec import load_sparse_matrix, save_recommender
    from mrec.sparse import fast_sparse_matrix

    file_format = sys.argv[1]
    filepath = sys.argv[2]
    outfile = sys.argv[3]

    # load training set as scipy sparse matrix
    train = load_sparse_matrix(file_format,filepath)

    model = WARPMFRecommender(d=100, gamma=0.01, C=100.0, batch_size=10, max_iters=7001, validation_iters = 1000, sample_item_rate=0.1)
    model.fit(train)

    # save_recommender(model,outfile)

if __name__ == '__main__':
    main()
