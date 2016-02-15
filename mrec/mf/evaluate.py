import random

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
        self.current_split = 0

    def __generate_current_split_path(self):
        return '%s/%d' % (self.split_dir, self.current_split)

    def __call__(self):
        from mrec import load_sparse_matrix
        import random
        current_split_path = self.__generate_current_split_path()
        train = load_sparse_matrix('csv', current_split_path)

        import warp

        max_iters,validation_iters,validation = warp.WARPMFRecommender.create_validation_set(train)
        users = validation.keys()
        return train, users, validation

def show_prec_history_for_model(prec_history, run_number, validation_iters):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 8))
    label = "abc"
    for model_name, prec_history_list in prec_history.items():
        iter_count = len(prec_history_list[0])
        average_history = [0] * iter_count
        for prec_history in prec_history_list:
            for i, prec in enumerate(prec_history):
                average_history[i] += prec
        average_history = map(lambda x: x/run_number, average_history)
        indexs = [i * validation_iters for i in range(iter_count) ]
        plt.plot(indexs, average_history, label=model_name.split("(")[0])
    plt.legend(loc='upper left', shadow=False)
    plt.xlabel("iteration count")
    plt.ylabel("prec@30")
    plt.title("WSabie Test")
    plt.show()
    pass

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
   #params = {'d':[100],'gamma':[0.01],'C':[100],'max_iters':[10000],'validation_iters':[2000], 'batch_size':[10],
   #          'model_type':['WARP', 'WARPLimitedItem'], 'sample_item_rate':[0.01], 'max_trials':[50]}
    params = {'d':[100],'gamma':[0.01],'C':[100],'max_iters':[1000],'validation_iters':[200], 'batch_size':[300],
              'model_type':['WARPLimitedItem'], 'sample_item_rate':[1], 'max_trials':[50]}
    models = [WARPMFRecommender(**a) for a in ParameterGrid(params)]

#   for train in glob:
#       # get test
#       # load em both up
#       # put them into something that returns train,test.keys(),test in a generator()
#       # test is a dict id->[id,id,...]

    if opts.main_split_dir:
        generate_main_metrics = generate_metrics(get_known_items_from_dict,compute_main_metrics)
        main_metrics, prec_history = run_evaluation(models,
                                      retrain_recommender,
                                      load_splits(opts.main_split_dir,opts.num_splits),
                                      opts.num_splits,
                                      generate_main_metrics)
        print_report(models,main_metrics)
        show_prec_history_for_model(prec_history, opts.num_splits, 1000)

    if opts.loo_split_dir:
        generate_hit_rate = generate_metrics(get_known_items_from_dict,compute_hit_rate)
        hit_rate_metrics = run_evaluation(models,
                                          retrain_recommender,
                                          load_splits(opts.loo_split_dir,opts.num_splits),
                                          opts.num_splits,
                                          generate_hit_rate)
        print_report(models,hit_rate_metrics)
