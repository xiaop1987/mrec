def retrain_recommender(model,dataset):
    model.fit(dataset.X)

class load_splits(object):
    '''
    Should return train_data,test_users,test_data

    Parameters
    ----------
    split_dir
    num_splits

    Returns
    -------

    '''
    def __init__(self, split_dir, num_splits):
        self.split_dir = split_dir
        self.num_splits = num_splits

    def __call__(self):
        from mrec import load_sparse_matrix
        import random
        train = load_sparse_matrix('csv', self.split_dir)

        num_users = train.shape[0]
        num_validation_users = max(num_users/100,10)
        # ensure reasonable expected number of updates per validation user
        validation_iters = 100*num_users/num_validation_users
        # and reasonable number of validation cycles
        max_iters = 30*validation_iters

        print num_validation_users,'validation users'
        print validation_iters,'validation iters'
        print max_iters,'max_iters'

        validation = dict()
        users = list()
        for u in xrange(num_validation_users):
            positive = np.where(train[u].data > 0)[0]
            hidden = random.sample(positive, positive.shape[0]/2)
            if hidden:
                train[u].data[hidden] = 0
                validation[u] = train[u].indices[hidden]
                users.append(u)
        return train, users, validation

if __name__ == '__main__':

    try:
        from sklearn.grid_search import ParameterGrid
    except ImportError:
        from sklearn.grid_search import IterGrid as ParameterGrid
    from optparse import OptionParser
    from warp import WARPMFRecommender

    from mrec.evaluation.metrics import *

    parser = OptionParser()
    parser.add_option('-m','--main_split_dir',dest='main_split_dir',help='directory containing 50/50 splits for main evaluation')
    parser.add_option('-l','--loo_split_dir',dest='loo_split_dir',help='directory containing LOO splits for hit rate evaluation')
    parser.add_option('-n','--num_splits',dest='num_splits',type='int',default=5,help='number of splits in each directory (default: %default)')

    (opts,args) = parser.parse_args()
    if not (opts.main_split_dir or opts.loo_split_dir) or not opts.num_splits:
        parser.print_help()
        raise SystemExit

    print 'Doing a grid search for regularization parameters...'
    params = {'d':[100],'gamma':[0.01],'C':[100],'max_iters':[20000],'validation_iters':[2000], 'batch_size':[1, 10]}
    models = [WARPMFRecommender(**a) for a in ParameterGrid(params)]

#   for train in glob:
#       # get test
#       # load em both up
#       # put them into something that returns train,test.keys(),test in a generator()
#       # test is a dict id->[id,id,...]

    if opts.main_split_dir:
        generate_main_metrics = generate_metrics(get_known_items_from_dict,compute_main_metrics)
        main_metrics = run_evaluation(models,
                                      retrain_recommender,
                                      load_splits(opts.main_split_dir,opts.num_splits),
                                      opts.num_splits,
                                      generate_main_metrics)
        print_report(models,main_metrics)

    if opts.loo_split_dir:
        generate_hit_rate = generate_metrics(get_known_items_from_dict,compute_hit_rate)
        hit_rate_metrics = run_evaluation(models,
                                          retrain_recommender,
                                          load_splits(opts.loo_split_dir,opts.num_splits),
                                          opts.num_splits,
                                          generate_hit_rate)
        print_report(models,hit_rate_metrics)
