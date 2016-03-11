import numpy as np
import random
from itertools import izip

from mrec.evaluation import metrics

import warp_fast
from warp_fast import warp_sample, apply_updates

class SampleOperate(object):
    def __init__(self, train, decomposition, positive_thresh, max_trials):
        self.train = train
        self.decomposition = decomposition
        self.positive_thresh = positive_thresh
        self.max_trials = max_trials
        self.selected_items = None

    def __call__(self, u = None):
        # delegate to cython implementation
        return warp_sample(self.decomposition.U,
                           self.decomposition.V,
                           self.train.data,
                           self.train.indices,
                           self.train.indptr,
                           self.positive_thresh,
                           self.max_trials,
                           self.selected_items,
                           u)

class SampleOperateWithLimitedItem(SampleOperate):
    def __init__(self, train, decomposition, positive_thresh, max_trials, sample_item_rate):
        super(SampleOperateWithLimitedItem, self).__init__(train, decomposition, positive_thresh, max_trials)
        user_count, item_count =  train.shape
        self.sample_item_rate = sample_item_rate
        self.selected_items = random.sample(range(0, item_count), int(item_count * sample_item_rate))

    def __call__(self, u = None):
        # delegate to cython implementation
       return warp_sample(self.decomposition.U,
                                     self.decomposition.V,
                                     self.train.data,
                                     self.train.indices,
                                     self.train.indptr,
                                     self.positive_thresh,
                                     self.max_trials,
                                     self.selected_items,
                                     u)

       #return u, i, j, N, trials #* self.sample_item_rate

class WARPBatchUpdate(object):
    """Collection of arrays to hold a batch of WARP sgd updates."""

    def __init__(self,batch_size,d):
        self.u = np.zeros(batch_size,dtype='int32')
        self.dU = np.zeros((batch_size,d),order='F')
        self.v_pos = np.zeros(batch_size,dtype='int32')
        self.dV_pos = np.zeros((batch_size,d))
        self.v_neg = np.zeros(batch_size,dtype='int32')
        self.dV_neg = np.zeros((batch_size,d))

    def clear(self):
        pass

    def set_update(self,ix,update):
        u,v_pos,v_neg,dU,dV_pos,dV_neg = update
        self.u[ix] = u
        self.dU[ix] = dU
        self.v_pos[ix] = v_pos
        self.dV_pos[ix] = dV_pos
        self.v_neg[ix] = v_neg
        self.dV_neg[ix] = dV_neg

class WARPDecomposition(object):
    """
    Matrix embedding optimizing the WARP loss.

    Parameters
    ==========
    num_rows : int
        Number of rows in the full matrix.
    num_cols : int
        Number of columns in the full matrix.
    d : int
        The embedding dimension for the decomposition.
    """

    def __init__(self,num_rows,num_cols,d):
        # initialize factors to small random values
        self.U = d**-0.5*np.random.random_sample((num_rows,d))
        self.V = d**-0.5*np.random.random_sample((num_cols,d))
        # ensure memory layout avoids extra allocation in dot product
        self.U = np.asfortranarray(self.U)

    def compute_gradient_step(self,u,i,j,L):
        """
        Compute a gradient step from results of sampling.

        Parameters
        ==========
        u : int
            The sampled row.
        i : int
            The sampled positive column.
        j : int
            The sampled violating negative column i.e. U[u].V[j] is currently
            too large compared to U[u].V[i]
        L : int
            The number of trials required to find a violating negative column.

        Returns
        =======
        u : int
            Sampled user id.
        i : int
            Postive item id.
        j : int
            Negative item id.
        dU : numpy.ndarray
            Gradient step for U[u].
        dV_pos : numpy.ndarray
            Gradient step for V[i].
        dV_neg : numpy.ndarray
            Gradient step for V[j].
        """
        dU, dV_pos, dV_neg = WARPDecomposition.compute_gradient_step_core(self.V[i], self.V[j], self.U[u], L)
        return u,i,j, dU, dV_pos, dV_neg

    @staticmethod
    def compute_gradient_step_core(positive_item_factor, negative_item_factor, user_factor, warp_loss):
        dU = warp_loss * (positive_item_factor- negative_item_factor)
        dV_pos = warp_loss * user_factor
        dV_neg = -warp_loss * user_factor
        return dU, dV_pos, dV_neg

    def apply_updates(self,updates,gamma,C):
        # delegate to cython implementation
        apply_updates(self.U,updates.u,updates.dU,gamma,C)
        apply_updates(self.V,updates.v_pos,updates.dV_pos,gamma,C)
        apply_updates(self.V,updates.v_neg,updates.dV_neg,gamma,C)

    def reconstruct(self,rows):
        if rows is None:
            U = self.U
        else:
            U = np.asfortranarray(self.U[rows,:])
        return U.dot(self.V.T)

class WARP(object):
    """
    Learn low-dimensional embedding optimizing the WARP loss.

    Parameters
    ==========
    d : int
        Embedding dimension.
    gamma : float
        Learning rate.
    C : float
        Regularization constant.
    max_iters : int
        Maximum number of SGD updates.
    validation_iters : int
        Number of SGD updates between checks for stopping condition.
    batch_size : int
        Mini batch size for SGD updates.
    positive_thresh: float
        Training entries below this are treated as zero.
    max_trials : int
        Number of attempts allowed to find a violating negative example during
        training updates. This means that in practice we optimize for ranks 1
        to max_trials-1.

    Attributes
    ==========
    U_ : numpy.ndarray
        Row factors.
    V_ : numpy.ndarray
        Column factors.
    """

    def __init__(self,
                 d,
                 gamma,
                 C,
                 max_iters,
                 validation_iters,
                 batch_size=10,
                 positive_thresh=0.00001,
                 max_trials=50):
        self.d = d
        self.gamma = gamma
        self.C = C
        self.max_iters = max_iters
        self.validation_iters = validation_iters
        self.batch_size = batch_size
        self.positive_thresh = positive_thresh
        self.max_trials = max_trials
        self.train_prec_history = []
        self.test_prec_history = []
        self.prec_history = {}

    def __str__(self):
        return 'WARP(d={0},gamma={1},C={2},max_iters={3},validation_iters={4},batch_size={5},positive_thresh={6},' \
               'max_trials={7})'.format(
                self.d,
                self.gamma,
                self.C,
                self.max_iters,
                self.validation_iters,
                self.batch_size,
                self.positive_thresh,
                self.max_trials
        )

    def recommend(self, users, topK):
      r = self.decomposition.reconstruct(users)
      prec = 0
      diff = 0
      sample_count = 0
      ret_dict = {}
      for u,ru in izip(users,r):
        predicted = ru.argsort()[::-1][:topK]
        ret_dict[u]=predicted
      return ret_dict

    def fit(self,train,validation=None):
        """
        Learn factors from training set. The dot product of the factors
        reconstructs the training matrix approximately, minimizing the
        WARP ranking loss relative to the original data.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix
            Training matrix to be factorized.
        validation : dict or int
            Validation set to control early stopping, based on precision@30.
            The dict should have the form row->[cols] where the values in cols
            are those we expected to be highly ranked in the reconstruction of
            row. If an int is supplied then instead we evaluate precision
            against the training data for the first validation rows.

        Returns
        =======
        self : object
            This model itself.
        """
        num_rows,num_cols = train.shape
        self.user_count = num_rows
        self.item_count = num_cols

        self.decomposition = WARPDecomposition(num_rows,num_cols,self.d)
        updates = WARPBatchUpdate(self.batch_size,self.d)
        self.warp_loss = self.precompute_warp_loss(num_cols)

        self._fit(self.decomposition,updates,train,validation)

        self.U_ = self.decomposition.U
        self.V_ = self.decomposition.V

        self.prec_history = {'train': self.train_prec_history,
                             'test': self.test_prec_history}

        return self

    def _fit(self,decomposition,updates,train,validation):
        self.train_prec_history = []
        self.test_prec_history = []
        tot_trials = 0
        for it in xrange(self.max_iters):
            if it % self.validation_iters == 0:
                print 'Average trials for each user: %f for iteration: %d' % (tot_trials/(float(self.batch_size) * self.validation_iters), it)
                tot_trials = 0
                train_prec, test_prec = self.estimate_precision(decomposition,train,validation)
                self.test_prec_history.append(test_prec)
                self.train_prec_history.append(train_prec)
                print '{0}: validation precision of train = {1:.3f}, of test = {2:.3f}'.format(it, train_prec, test_prec)
               #if len(precs) > 10 and precs[-1] < precs[-2] and precs[-2] < precs[-3]:
               #    print 'validation precision got worse twice, terminating'
               #    break
            tot_trials += self.compute_updates(train,decomposition,updates)
            decomposition.apply_updates(updates,self.gamma,self.C)

    @staticmethod
    def precompute_warp_loss(num_cols):
        """
        Precompute WARP loss for each possible rank:

            L(i) = \sum_{0,i}{1/(i+1)}
        """
        assert(num_cols>1)
        warp_loss = np.ones(num_cols)
        for i in xrange(1,num_cols):
            warp_loss[i] = warp_loss[i-1]+1.0/(i+1)
        return warp_loss

    @staticmethod
    def estimate_warp_loss(train,u,N, warp_loss):
        num_cols = train.shape[1]
        nnz = train.indptr[u+1]-train.indptr[u]
        estimated_rank = (num_cols-nnz-1)/N
        return WARP.estimate_warp_loss_core(num_cols, nnz, N, warp_loss)

    @staticmethod
    def estimate_warp_loss_core(total_item_count, current_user_count, trials, warp_loss):
        estimated_rank = (total_item_count-current_user_count-1)/trials
        return warp_loss[estimated_rank]

    def compute_updates(self,train,decomposition,updates):
        updates.clear()
        tot_trials = 0

        sample_operate = self.generate_sample_operate(train, decomposition, self.positive_thresh, self.max_trials)
        for ix in xrange(self.batch_size):
            ret = sample_operate()
            if ret == None:
                continue
            u,i,j,N,trials = ret
            tot_trials += trials
            L = WARP.estimate_warp_loss(train,u,N, self.warp_loss)
            updates.set_update(ix,decomposition.compute_gradient_step(u,i,j,L))
       #sample_user_list = []
       #for i in xrange(self.batch_size):
       #    sample_user_list.append(random.randint(0, self.user_count - 1))

       #update_user_count = 0.0
       #for ix, u in enumerate(sample_user_list):
       #    ret = sample_operate(u)
       #    if ret == None:
       #        continue
       #    update_user_count += 1
       #    u,i,j,N,trials = ret
       #    tot_trials += trials
       #    L = WARP.estimate_warp_loss(train,u,N, self.warp_loss)
       #    updates.set_update(ix,decomposition.compute_gradient_step(u,i,j,L))
       #print "%f percent of user updated" % (update_user_count / self.batch_size)
        return tot_trials

    def generate_sample_operate(self, train, decomposition, positive_thresh, max_trials):
        return SampleOperate(train, decomposition, positive_thresh, max_trials)

    def estimate_precision(self,decomposition,train,validation,k=150):
        """
        Compute prec@k for a sample of training rows.

        Parameters
        ==========
        decomposition : WARPDecomposition
            The current decomposition.
        train : scipy.sparse.csr_matrix
            The training data.
        k : int
            Measure precision@k.
        validation : dict or int
            Validation set over which we compute precision. Either supply
            a dict of row -> list of hidden cols, or an integer n, in which
            case we simply evaluate against the training data for the first
            n rows.

        Returns
        =======
        prec : float
            Precision@k computed over a sample of the training rows.

        Notes
        =====
        At the moment this will underestimate the precision of real
        recommendations because we do not exclude training cols with zero
        ratings from the top-k predictions evaluated.
        """

        rows = validation.keys()
        r = decomposition.reconstruct(rows)
        train_prec = 0
        test_prec = 0
        diff = 0
        sample_count = 0
        for u,ru in izip(rows,r):
            predicted = ru.argsort()[::-1][:k]
            actual = validation[u]
            train_data = actual['train']
            test_data = actual['test']
            train_prec += metrics.prec(predicted,train_data,k, True)
            test_prec += metrics.prec(predicted,test_data,k, True)
        return float(train_prec)/len(rows), float(test_prec)/len(rows)

class WARPLimitedItem(WARP):
    def __init__(self,
                 d,
                 gamma,
                 C,
                 max_iters,
                 validation_iters,
                 batch_size=10,
                 positive_thresh=0.00001,
                 max_trials=50,
                 sample_item_rate = 1.0):
        super(WARPLimitedItem, self).__init__(d, gamma, C, max_iters, validation_iters, batch_size, positive_thresh)
        self.sample_item_rate = sample_item_rate

    def generate_sample_operate(self, train, decomposition, positive_thresh, max_trials):
        return SampleOperateWithLimitedItem(train, decomposition, positive_thresh, max_trials, self.sample_item_rate)

    def __str__(self):
        return 'WARPSampleItem(d={0},gamma={1},C={2},max_iters={3},validation_iters={4},batch_size={5},positive_thresh={6},' \
               'max_trials={7}, item_sample_rate = {8})'.format(
                self.d,
                self.gamma,
                self.C,
                self.max_iters,
                self.validation_iters,
                self.batch_size,
                self.positive_thresh,
                self.max_trials,
                self.sample_item_rate
        )

